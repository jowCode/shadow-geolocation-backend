"""
Bundle Adjustment für Shadow Geolocation

VERSION 2.0:
- Echtes FOV statt master_focal_length
- Koordinatensystem mit Ursprung bei (0, 0, 0)
- Globale Kamera-Position (nicht pro Screenshot)
- Error-Signal: backgroundOffset != (50, 50)

KOORDINATENSYSTEM:
- Ursprung: Linke untere Ecke der Front-Wand
- X: Nach rechts (0 → room.width)
- Y: Nach oben (0 → room.height)
- Z: Nach hinten (0 → room.depth)
- Einheit: Meter
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, AsyncGenerator, Optional
import asyncio
from dataclasses import dataclass


@dataclass
class CalibrationScreenshot:
    """Ein kalibrierter Screenshot"""
    id: str
    # Kamera-Blickrichtung (Rotation) für diesen Screenshot
    camera_rotation: Dict[str, float]  # {x, y, z} in Grad
    # Display-Parameter (Error-Signal!)
    background_offset_x: float  # Prozent (50 = zentriert)
    background_offset_y: float  # Prozent (50 = zentriert)
    background_rotation: float  # Grad (0 = keine Korrektur)
    background_scale: float     # Prozent (nur UI)
    completed: bool


@dataclass
class CalibrationData:
    """Vollständige Kalibrierungs-Daten (v2.0)"""
    room: Dict[str, float]           # {width, depth, height} in Metern
    camera_position: Dict[str, float] # {x, y, z} in Metern (GLOBAL!)
    fov_y: float                     # Vertikales FOV in Grad
    screenshots: List[CalibrationScreenshot]


def rotation_matrix_yxz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """
    Erstelle Rotationsmatrix aus Euler-Winkeln (in Grad).
    Rotation Order: YXZ (wie Three.js)
    
    - Y (Yaw): Schwenk links/rechts
    - X (Pitch): Neigung hoch/runter
    - Z (Roll): Rotation um Blickachse
    """
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)
    z = np.deg2rad(z_deg)
    
    # Rotation um Y-Achse (Yaw)
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    # Rotation um X-Achse (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    # Rotation um Z-Achse (Roll)
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # YXZ Order: Y zuerst, dann X, dann Z
    return Rz @ Rx @ Ry


def project_point_to_2d(
    point_3d: Tuple[float, float, float],
    camera_pos: Tuple[float, float, float],
    camera_rotation: Dict[str, float],
    fov_y: float,
    aspect_ratio: float = 16/9
) -> Tuple[float, float]:
    """
    Projiziere einen 3D-Punkt auf 2D-Bildkoordinaten.
    
    Args:
        point_3d: 3D-Punkt im Welt-Koordinatensystem (Meter)
        camera_pos: Kamera-Position (Meter)
        camera_rotation: Kamera-Blickrichtung {x, y, z} in Grad
        fov_y: Vertikales Field of View in Grad
        aspect_ratio: Breite/Höhe des Bildes
    
    Returns:
        (x_percent, y_percent) - Position in Bild-Koordinaten (0-100%)
        (50, 50) = Bildmitte
    """
    # 1. Translation: Welt → Kamera-relativ
    cam = np.array(camera_pos)
    point = np.array(point_3d)
    relative = point - cam
    
    # 2. Rotation: Welt → Kamera-Koordinaten
    R = rotation_matrix_yxz(
        camera_rotation['x'],
        camera_rotation['y'],
        camera_rotation['z']
    )
    # Inverse Rotation (Kamera-Transformation)
    camera_coords = R.T @ relative
    
    # In Kamera-Koordinaten:
    # X = rechts, Y = oben, Z = Blickrichtung (negativ = vor Kamera)
    
    # 3. Prüfe ob Punkt vor der Kamera ist
    if camera_coords[2] <= 0:
        # Hinter der Kamera → projiziere trotzdem für Gradient
        # (aber mit großem Error)
        camera_coords[2] = 0.001
    
    # 4. Perspektivische Projektion mit echtem FOV
    # tan(fov/2) = (Bildhöhe/2) / Brennweite
    tan_half_fov = np.tan(np.deg2rad(fov_y / 2))
    
    # Normalisierte Bildkoordinaten (-1 bis +1)
    ndc_x = camera_coords[0] / (camera_coords[2] * tan_half_fov * aspect_ratio)
    ndc_y = camera_coords[1] / (camera_coords[2] * tan_half_fov)
    
    # 5. Konvertiere zu Prozent (0-100, Mitte = 50)
    x_percent = 50 + ndc_x * 50
    y_percent = 50 - ndc_y * 50  # Y invertiert (Bild-Koordinaten)
    
    return (x_percent, y_percent)


def project_room_center_to_2d(
    room_dims: Tuple[float, float, float],
    camera_pos: Tuple[float, float, float],
    camera_rotation: Dict[str, float],
    fov_y: float
) -> Tuple[float, float]:
    """
    Projiziere das Raum-Zentrum auf 2D-Bildkoordinaten.
    
    Das Raum-Zentrum ist der Referenzpunkt für die Kalibrierung.
    Wenn alles korrekt kalibriert ist, sollte es bei (50, 50) erscheinen.
    """
    room_center = (
        room_dims[0] / 2,  # X = Mitte der Breite
        room_dims[1] / 2,  # Y = Mitte der Höhe
        room_dims[2] / 2   # Z = Mitte der Tiefe
    )
    
    return project_point_to_2d(
        room_center,
        camera_pos,
        camera_rotation,
        fov_y
    )


def compute_projection_error(
    params: np.ndarray,
    screenshots: List[CalibrationScreenshot],
    fov_y: float,
    initial_room: Dict,
    initial_camera: Dict,
    weights: Dict
) -> float:
    """
    Berechne den Projektions-Fehler.
    
    Error-Signal:
    1. backgroundOffset != (50, 50) bedeutet:
       - Die projizierten Raum-Linien passen nicht zum Screenshot
       - User musste den Screenshot verschieben
    
    2. Regularization:
       - Abweichung von initialen Werten (gewichtet nach Confidence)
    
    Args:
        params: [room_width, room_depth, room_height, cam_x, cam_y, cam_z]
        screenshots: Kalibrierte Screenshots
        fov_y: Field of View (fix, nicht optimiert)
        initial_room: Initiale Raum-Dimensionen
        initial_camera: Initiale Kamera-Position
        weights: {room_confidence, position_confidence}
    
    Returns:
        Gewichteter Gesamtfehler
    """
    # Parameter extrahieren
    room_width, room_depth, room_height = params[0:3]
    cam_x, cam_y, cam_z = params[3:6]
    
    # Reprojection Error
    reprojection_error = 0.0
    screenshot_count = 0
    
    for screenshot in screenshots:
        if not screenshot.completed:
            continue
        
        screenshot_count += 1
        
        # Projiziere Raum-Zentrum mit aktuellen Parametern
        projected_x, projected_y = project_room_center_to_2d(
            room_dims=(room_width, room_height, room_depth),
            camera_pos=(cam_x, cam_y, cam_z),
            camera_rotation=screenshot.camera_rotation,
            fov_y=fov_y
        )
        
        # Error: Wie weit ist die Projektion vom aktuellen Offset entfernt?
        # Wenn Offset = (50, 50) und Projektion = (50, 50) → Error = 0
        # Wenn Offset = (60, 40) und Projektion = (50, 50) → Error groß
        
        # Das Offset zeigt, wohin der User den Screenshot verschieben musste
        # Die Projektion zeigt, wo das Raum-Zentrum erscheinen würde
        
        # Ideal: projected == offset (User musste nicht verschieben)
        # Real: Differenz zwischen "wo das Zentrum projiziert wird" und "wo der User es hingeschoben hat"
        
        offset_error_x = screenshot.background_offset_x - projected_x
        offset_error_y = screenshot.background_offset_y - projected_y
        
        reprojection_error += offset_error_x**2 + offset_error_y**2
    
    # Normalisieren
    if screenshot_count > 0:
        reprojection_error /= screenshot_count
    
    # Regularization: Wie weit weichen wir vom Initial Guess ab?
    
    # Raum: Normalisiert auf Prozent-Abweichung
    room_deviation = (
        ((room_width - initial_room['width']) / max(initial_room['width'], 0.1))**2 +
        ((room_depth - initial_room['depth']) / max(initial_room['depth'], 0.1))**2 +
        ((room_height - initial_room['height']) / max(initial_room['height'], 0.1))**2
    )
    
    # Position: Normalisiert auf Raum-Dimensionen
    position_deviation = (
        ((cam_x - initial_camera['x']) / max(initial_room['width'], 0.1))**2 +
        ((cam_y - initial_camera['y']) / max(initial_room['height'], 0.1))**2 +
        ((cam_z - initial_camera['z']) / max(initial_room['depth'], 0.1))**2
    )
    
    # Gewichte: Hohe Confidence = starke Regularization
    # (User ist sicher → Parameter soll nicht zu weit abweichen)
    room_weight = weights.get('room_confidence', 0.5) * 10
    position_weight = weights.get('position_confidence', 0.5) * 10
    
    # Gewichtete Summe
    total_error = (
        reprojection_error +
        room_weight * room_deviation +
        position_weight * position_deviation
    )
    
    return total_error


async def bundle_adjustment_async(
    calibration_data: CalibrationData,
    weights: Dict = None,
) -> AsyncGenerator[Dict, None]:
    """
    Führe Bundle Adjustment durch mit Progress-Updates.
    
    VERSION 2.0:
    - Optimiert Raum-Dimensionen und globale Kamera-Position
    - Verwendet echtes FOV (nicht master_focal_length)
    - Error-Signal: backgroundOffset != (50, 50)
    
    Yields:
        Progress-Updates als Dict
    """
    
    if weights is None:
        weights = {
            'room_confidence': 0.5,
            'position_confidence': 0.5
        }
    
    try:
        yield {
            "type": "progress",
            "progress": 0,
            "message": "Starte Bundle Adjustment (v2.0)...",
            "iteration": 0
        }
        
        await asyncio.sleep(0.1)
        
        # Nur completed Screenshots verwenden
        completed_screenshots = [s for s in calibration_data.screenshots if s.completed]
        
        if len(completed_screenshots) < 2:
            yield {
                "type": "error",
                "message": "Mindestens 2 kalibrierte Screenshots erforderlich!"
            }
            return
        
        # Initial Guess
        room = calibration_data.room
        initial_camera = calibration_data.camera_position
        fov_y = calibration_data.fov_y
        
        print(f"📐 Bundle Adjustment v2.0")
        print(f"   Raum: {room['width']:.1f} x {room['depth']:.1f} x {room['height']:.1f}m")
        print(f"   Kamera: ({initial_camera['x']:.2f}, {initial_camera['y']:.2f}, {initial_camera['z']:.2f})")
        print(f"   FOV: {fov_y}°")
        print(f"   Screenshots: {len(completed_screenshots)}")
        print(f"   Weights: {weights}")
        
        yield {
            "type": "progress",
            "progress": 10,
            "message": f"Optimiere mit {len(completed_screenshots)} Screenshots...",
            "iteration": 0
        }
        
        await asyncio.sleep(0.1)
        
        # Start-Werte
        x0 = np.array([
            room['width'],
            room['depth'],
            room['height'],
            initial_camera['x'],
            initial_camera['y'],
            initial_camera['z']
        ])
        
        # Bounds (physikalisch plausibel)
        bounds = [
            (2.0, 20.0),   # Raum-Breite: 2-20m
            (2.0, 20.0),   # Raum-Tiefe: 2-20m
            (2.0, 6.0),    # Raum-Höhe: 2-6m
            (0.1, room['width'] - 0.1),   # Kamera X: innerhalb Raum
            (0.1, room['height'] - 0.1),  # Kamera Y: innerhalb Raum
            (0.1, room['depth'] - 0.1)    # Kamera Z: innerhalb Raum
        ]
        
        # Iterations-Zähler
        iteration_count = [0]
        
        def callback(xk):
            iteration_count[0] += 1
        
        yield {
            "type": "progress",
            "progress": 20,
            "message": "Optimierung läuft...",
            "iteration": 0
        }
        
        await asyncio.sleep(0.1)
        
        # Optimierung
        result = minimize(
            fun=compute_projection_error,
            x0=x0,
            args=(completed_screenshots, fov_y, room, initial_camera, weights),
            bounds=bounds,
            method='L-BFGS-B',
            callback=callback,
            options={
                'maxiter': 100,
                'ftol': 1e-6
            }
        )
        
        yield {
            "type": "progress",
            "progress": 90,
            "message": "Bereite Ergebnisse vor...",
            "iteration": iteration_count[0]
        }
        
        await asyncio.sleep(0.1)
        
        # Ergebnis aufbereiten
        optimized_room = {
            'width': float(result.x[0]),
            'depth': float(result.x[1]),
            'height': float(result.x[2])
        }
        
        optimized_camera = {
            'x': float(result.x[3]),
            'y': float(result.x[4]),
            'z': float(result.x[5])
        }
        
        # Error-Berechnung
        initial_error = compute_projection_error(
            x0, completed_screenshots, fov_y, room, initial_camera, weights
        )
        final_error = result.fun
        
        improvement_percent = ((initial_error - final_error) / max(initial_error, 0.001)) * 100
        
        # Durchschnittlicher Offset-Error (für Diagnostik)
        avg_offset_error = 0.0
        for s in completed_screenshots:
            proj_x, proj_y = project_room_center_to_2d(
                (optimized_room['width'], optimized_room['height'], optimized_room['depth']),
                (optimized_camera['x'], optimized_camera['y'], optimized_camera['z']),
                s.camera_rotation,
                fov_y
            )
            avg_offset_error += abs(s.background_offset_x - proj_x) + abs(s.background_offset_y - proj_y)
        avg_offset_error /= len(completed_screenshots)
        
        print(f"✅ Optimierung abgeschlossen:")
        print(f"   Error: {initial_error:.4f} → {final_error:.4f} ({improvement_percent:.1f}%)")
        print(f"   Raum: {optimized_room['width']:.2f} x {optimized_room['depth']:.2f} x {optimized_room['height']:.2f}m")
        print(f"   Kamera: ({optimized_camera['x']:.2f}, {optimized_camera['y']:.2f}, {optimized_camera['z']:.2f})")
        print(f"   Avg Offset Error: {avg_offset_error:.1f}%")
        
        yield {
            "type": "result",
            "progress": 100,
            "message": "Bundle Adjustment erfolgreich!",
            "result": {
                'optimized_room': optimized_room,
                'optimized_camera': optimized_camera,
                'optimized_fov': fov_y,  # FOV war fix
                'initial_error': float(initial_error),
                'final_error': float(final_error),
                'improvement_percent': float(improvement_percent),
                'iterations': int(iteration_count[0]),
                'success': bool(result.success),
                'message': str(result.message) if hasattr(result, 'message') else '',
                'avg_offset_error': float(avg_offset_error),
                # Legacy-Felder für Frontend-Kompatibilität
                'positions_variance_before': 0.0,
                'positions_variance_after': 0.0,
                'variance_reduction_percent': 0.0
            }
        }
        
    except Exception as e:
        print(f"❌ Bundle Adjustment Fehler: {str(e)}")
        import traceback
        traceback.print_exc()
        yield {
            "type": "error",
            "message": f"Bundle Adjustment Fehler: {str(e)}"
        }


# ============================================================================
# ADAPTER-FUNKTION für Legacy-Format
# ============================================================================

def convert_legacy_request(raw_data: dict) -> CalibrationData:
    """
    Konvertiert Legacy-Request-Format zu CalibrationData.
    
    Unterstützt sowohl v1.0 (master_focal_length) als auch v2.0 (camera.fov_y)
    """
    
    # Raum
    room = raw_data['room']
    
    # Kamera-Position
    if 'camera' in raw_data and raw_data['camera']:
        camera_position = raw_data['camera'].get('position', raw_data.get('global_camera_position', {'x': 2, 'y': 1.5, 'z': 0.5}))
        fov_y = raw_data['camera'].get('fov_y', raw_data.get('global_fov_y', 60))
    else:
        camera_position = raw_data.get('global_camera_position', {'x': 2, 'y': 1.5, 'z': 0.5})
        fov_y = raw_data.get('global_fov_y', 60)
    
    # Wenn weder v2.0 noch legacy FOV vorhanden, benutze Default
    if fov_y is None:
        fov_y = 60
    
    # Screenshots konvertieren
    screenshots = []
    for s in raw_data['screenshots']:
        # Kamera-Rotation (v2.0 oder Legacy)
        if 'camera_rotation' in s and s['camera_rotation']:
            camera_rotation = {
                'x': s['camera_rotation'].get('x', 0),
                'y': s['camera_rotation'].get('y', 0),
                'z': s['camera_rotation'].get('z', 0)
            }
        elif 'room_rotation' in s and s['room_rotation']:
            camera_rotation = s['room_rotation']
        else:
            camera_rotation = {'x': 0, 'y': 0, 'z': 0}
        
        # Display-Parameter (v2.0 oder Legacy)
        if 'display' in s and s['display']:
            display = s['display']
            background_offset_x = display.get('background_offset_x', 50)
            background_offset_y = display.get('background_offset_y', 50)
            background_rotation = display.get('background_rotation', 0)
            background_scale = display.get('background_scale', 50)
        else:
            background_offset_x = s.get('background_offset_x', 50)
            background_offset_y = s.get('background_offset_y', 50)
            background_rotation = s.get('background_rotation', 0)
            background_scale = s.get('background_scale', 50)
        
        screenshots.append(CalibrationScreenshot(
            id=s['id'],
            camera_rotation=camera_rotation,
            background_offset_x=background_offset_x,
            background_offset_y=background_offset_y,
            background_rotation=background_rotation,
            background_scale=background_scale,
            completed=s.get('completed', False)
        ))
    
    return CalibrationData(
        room=room,
        camera_position=camera_position,
        fov_y=fov_y,
        screenshots=screenshots
    )