"""
Bundle Adjustment für Shadow Geolocation

VERSION 3.0:
- Verwendet neues CalibrationData Format
- Kein Legacy v1/v2 Support
- Klare Struktur

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


def rotation_matrix_yxz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """
    Erstelle Rotationsmatrix aus Euler-Winkeln (in Grad).
    Rotation Order: YXZ (wie Three.js)
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
    
    # YXZ Order
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
    
    Returns:
        (x_percent, y_percent) - Position in Bild-Koordinaten (0-100%)
        (50, 50) = Bildmitte
    """
    cam = np.array(camera_pos)
    point = np.array(point_3d)
    relative = point - cam
    
    R = rotation_matrix_yxz(
        camera_rotation.get('x', 0),
        camera_rotation.get('y', 0),
        camera_rotation.get('z', 0)
    )
    camera_coords = R.T @ relative
    
    if camera_coords[2] <= 0:
        camera_coords[2] = 0.001
    
    tan_half_fov = np.tan(np.deg2rad(fov_y / 2))
    
    ndc_x = camera_coords[0] / (camera_coords[2] * tan_half_fov * aspect_ratio)
    ndc_y = camera_coords[1] / (camera_coords[2] * tan_half_fov)
    
    x_percent = 50 + ndc_x * 50
    y_percent = 50 - ndc_y * 50
    
    return (x_percent, y_percent)


def project_room_center_to_2d(
    room_dims: Tuple[float, float, float],
    camera_pos: Tuple[float, float, float],
    camera_rotation: Dict[str, float],
    fov_y: float
) -> Tuple[float, float]:
    """Projiziere das Raum-Zentrum auf 2D-Bildkoordinaten."""
    room_center = (
        room_dims[0] / 2,
        room_dims[1] / 2,
        room_dims[2] / 2
    )
    
    return project_point_to_2d(
        room_center,
        camera_pos,
        camera_rotation,
        fov_y
    )


def compute_projection_error(
    params: np.ndarray,
    screenshots: List[Dict],
    fov_y: float,
    initial_room: Dict,
    initial_camera: Dict,
    weights: Dict
) -> float:
    """
    Berechne den Projektions-Fehler.
    
    Args:
        params: [room_width, room_depth, room_height, cam_x, cam_y, cam_z]
        screenshots: Kalibrierte Screenshots (v3.0 Format)
        fov_y: Field of View (fix)
        initial_room: Initiale Raum-Dimensionen
        initial_camera: Initiale Kamera-Position
        weights: {room_confidence, position_confidence}
    """
    room_width, room_depth, room_height = params[0:3]
    cam_x, cam_y, cam_z = params[3:6]
    
    # Reprojection Error
    reprojection_error = 0.0
    screenshot_count = 0
    
    for screenshot in screenshots:
        if not screenshot.get('completed', False):
            continue
        
        screenshot_count += 1
        
        # Kamera-Rotation aus v3.0 Format
        camera_rotation = screenshot.get('cameraRotation', {'x': 0, 'y': 0, 'z': 0})
        
        # Display-Offset aus v3.0 Format
        display = screenshot.get('display', {})
        offset_x = display.get('backgroundOffsetX', 50)
        offset_y = display.get('backgroundOffsetY', 50)
        
        projected_x, projected_y = project_room_center_to_2d(
            room_dims=(room_width, room_height, room_depth),
            camera_pos=(cam_x, cam_y, cam_z),
            camera_rotation=camera_rotation,
            fov_y=fov_y
        )
        
        offset_error_x = offset_x - projected_x
        offset_error_y = offset_y - projected_y
        
        reprojection_error += offset_error_x**2 + offset_error_y**2
    
    if screenshot_count > 0:
        reprojection_error /= screenshot_count
    
    # Regularization
    room_width_deviation = (room_width - initial_room['width']) / initial_room['width']
    room_depth_deviation = (room_depth - initial_room['depth']) / initial_room['depth']
    room_height_deviation = (room_height - initial_room['height']) / initial_room['height']
    
    room_deviation = (
        room_width_deviation**2 +
        room_depth_deviation**2 +
        room_height_deviation**2
    )
    
    cam_x_deviation = (cam_x - initial_camera['x']) / initial_room['width']
    cam_y_deviation = (cam_y - initial_camera['y']) / initial_room['height']
    cam_z_deviation = (cam_z - initial_camera['z']) / initial_room['depth']
    
    position_deviation = (
        cam_x_deviation**2 +
        cam_y_deviation**2 +
        cam_z_deviation**2
    )
    
    room_confidence = weights.get('room_confidence', 0.5)
    position_confidence = weights.get('position_confidence', 0.5)
    
    base_room_weight = 50.0
    base_position_weight = 20.0
    
    room_weight = base_room_weight + room_confidence * 200.0
    position_weight = base_position_weight + position_confidence * 100.0
    
    total_error = (
        reprojection_error +
        room_weight * room_deviation +
        position_weight * position_deviation
    )
    
    return total_error


async def bundle_adjustment_async(
    calibration_data: Dict,
    weights: Dict = None,
) -> AsyncGenerator[Dict, None]:
    """
    Führe Bundle Adjustment durch mit Progress-Updates.
    
    VERSION 3.0:
    - Erwartet neues CalibrationData Format
    - Kein Legacy-Support
    
    Args:
        calibration_data: CalibrationData als Dict (v3.0 Format)
        weights: {room_confidence, position_confidence}
    
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
            "message": "Starte Bundle Adjustment (v3.0)...",
            "iteration": 0
        }
        
        await asyncio.sleep(0.1)
        
        # Extrahiere Daten aus v3.0 Format
        room = calibration_data.get('room', {})
        camera = calibration_data.get('camera', {})
        camera_position = camera.get('position', {'x': 2, 'y': 1.5, 'z': 0.5})
        fov_y = camera.get('fovY', 60)
        screenshots = calibration_data.get('screenshots', [])
        
        # Nur completed Screenshots
        completed_screenshots = [s for s in screenshots if s.get('completed', False)]
        
        if len(completed_screenshots) < 2:
            yield {
                "type": "error",
                "message": "Mindestens 2 kalibrierte Screenshots erforderlich!"
            }
            return
        
        print(f"📐 Bundle Adjustment v3.0")
        print(f"   Raum: {room.get('width', 0):.1f} x {room.get('depth', 0):.1f} x {room.get('height', 0):.1f}m")
        print(f"   Kamera: ({camera_position.get('x', 0):.2f}, {camera_position.get('y', 0):.2f}, {camera_position.get('z', 0):.2f})")
        print(f"   FOV: {fov_y}°")
        print(f"   Screenshots: {len(completed_screenshots)}")
        
        yield {
            "type": "progress",
            "progress": 10,
            "message": f"Optimiere mit {len(completed_screenshots)} Screenshots...",
            "iteration": 0
        }
        
        await asyncio.sleep(0.1)
        
        # Start-Werte
        initial_room = room
        initial_camera = camera_position
        
        x0 = np.array([
            room.get('width', 5),
            room.get('depth', 5),
            room.get('height', 3),
            camera_position.get('x', 2),
            camera_position.get('y', 1.5),
            camera_position.get('z', 0.5)
        ])
        
        # Dynamische Bounds
        room_confidence = weights.get('room_confidence', 0.5)
        position_confidence = weights.get('position_confidence', 0.5)
        
        room_max_deviation = 0.5 - room_confidence * 0.3
        pos_max_deviation = 0.5 - position_confidence * 0.3
        
        bounds = [
            (max(2.0, room.get('width', 5) * (1 - room_max_deviation)),
             room.get('width', 5) * (1 + room_max_deviation)),
            (max(2.0, room.get('depth', 5) * (1 - room_max_deviation)),
             room.get('depth', 5) * (1 + room_max_deviation)),
            (max(2.0, room.get('height', 3) * (1 - room_max_deviation)),
             min(6.0, room.get('height', 3) * (1 + room_max_deviation))),
            (max(0.1, camera_position.get('x', 2) - room.get('width', 5) * pos_max_deviation),
             min(room.get('width', 5) - 0.1, camera_position.get('x', 2) + room.get('width', 5) * pos_max_deviation)),
            (max(0.1, camera_position.get('y', 1.5) - room.get('height', 3) * pos_max_deviation),
             min(room.get('height', 3) - 0.1, camera_position.get('y', 1.5) + room.get('height', 3) * pos_max_deviation)),
            (max(0.1, camera_position.get('z', 0.5) - room.get('depth', 5) * pos_max_deviation),
             min(room.get('depth', 5) - 0.1, camera_position.get('z', 0.5) + room.get('depth', 5) * pos_max_deviation))
        ]
        
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
            args=(completed_screenshots, fov_y, initial_room, initial_camera, weights),
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
        
        # Ergebnis (v3.0 Format)
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
        
        initial_error = compute_projection_error(
            x0, completed_screenshots, fov_y, initial_room, initial_camera, weights
        )
        final_error = result.fun
        
        improvement_percent = ((initial_error - final_error) / max(initial_error, 0.001)) * 100
        
        print(f"✅ Optimierung abgeschlossen:")
        print(f"   Error: {initial_error:.4f} → {final_error:.4f} ({improvement_percent:.1f}%)")
        print(f"   Raum: {room.get('width', 0):.2f} → {optimized_room['width']:.2f}m")
        
        yield {
            "type": "result",
            "progress": 100,
            "message": "Bundle Adjustment erfolgreich!",
            "result": {
                'optimized_room': optimized_room,
                'optimized_camera': optimized_camera,
                'initial_error': float(initial_error),
                'final_error': float(final_error),
                'improvement_percent': float(improvement_percent),
                'iterations': int(iteration_count[0]),
                'success': bool(result.success),
                'message': str(result.message) if hasattr(result, 'message') else ''
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
