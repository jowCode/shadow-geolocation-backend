"""
Validation Solver für Shadow Geolocation

Validiert die Konsistenz der markierten Schatten-Punkte.

GEOMETRIE:
- Objekt-Punkt: 2D im Bild → 3D-Strahl von Kamera
- Schatten-Punkt: 2D im Bild + Wand → konkreter 3D-Punkt auf Wand
- Lichtrichtung: Vektor von Objekt zu Schatten (parallel für Sonnenlicht)

VALIDIERUNG:
1. Intra-Objekt: Alle 3 Punkt-Paare eines Objekts → eine Lichtrichtung?
2. Inter-Objekt: Alle Objekte eines Screenshots → dieselbe Lichtrichtung?
3. Global: Konsistenz über alle Screenshots (mit Zeitversatz)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'Point3D':
        return Point3D(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class Ray3D:
    origin: np.ndarray
    direction: np.ndarray  # normalized


@dataclass
class ValidationResult:
    success: bool
    status: str  # 'valid', 'warning', 'error', 'pending'
    consistency_score: float  # 0-100
    light_direction: Optional[Point3D]
    average_error_deg: float
    max_error_deg: float
    message: str
    details: Dict


# =============================================================================
# COORDINATE TRANSFORMS
# =============================================================================

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


def unproject_2d_to_ray(
    normalized_x: float,
    normalized_y: float,
    camera_position: Dict,
    camera_rotation: Dict,
    fov_y: float,
    aspect_ratio: float = 16/9
) -> Ray3D:
    """
    Konvertiere 2D-Bildpunkt zu 3D-Strahl.
    
    Args:
        normalized_x, normalized_y: 0-1, wobei (0,0) = oben-links
        camera_position: {x, y, z}
        camera_rotation: {x, y, z} in Grad
        fov_y: vertikales FOV in Grad
        aspect_ratio: Breite/Höhe
    
    Returns:
        Ray3D mit Origin (Kamera) und normalisierter Richtung
    """
    # Konvertiere zu NDC (-1 bis +1)
    # normalized 0-1 → NDC: x: 0→-1, 1→+1; y: 0→+1, 1→-1 (y invertiert)
    ndc_x = normalized_x * 2 - 1
    ndc_y = -(normalized_y * 2 - 1)  # Y invertieren (Bild-Y nach unten, 3D-Y nach oben)
    
    # FOV zu Tangens
    tan_half_fov = np.tan(np.deg2rad(fov_y / 2))
    
    # Richtung in Kamera-Koordinaten (Kamera schaut in +Z)
    dir_camera = np.array([
        ndc_x * tan_half_fov * aspect_ratio,
        ndc_y * tan_half_fov,
        1.0
    ])
    
    # Normalisieren
    dir_camera = dir_camera / np.linalg.norm(dir_camera)
    
    # Rotation anwenden (Kamera → Welt)
    R = rotation_matrix_yxz(
        camera_rotation.get('x', 0),
        camera_rotation.get('y', 0),
        camera_rotation.get('z', 0)
    )
    dir_world = R @ dir_camera
    
    origin = np.array([
        camera_position['x'],
        camera_position['y'],
        camera_position['z']
    ])
    
    return Ray3D(origin=origin, direction=dir_world / np.linalg.norm(dir_world))


def get_wall_plane(wall_name: str, room: Dict) -> Tuple[np.ndarray, float]:
    """
    Gibt Ebenen-Normal und d für Ebenengleichung n·p + d = 0 zurück.
    
    KOORDINATENSYSTEM:
    - Ursprung: Linke untere Ecke der Front-Wand
    - X: Nach rechts (0 → room.width)
    - Y: Nach oben (0 → room.height)
    - Z: Nach hinten (0 → room.depth)
    """
    w = room['width']
    h = room['height']
    d = room['depth']
    
    if wall_name == 'back':
        # z = depth
        return np.array([0, 0, 1]), -d
    elif wall_name == 'front':
        # z = 0
        return np.array([0, 0, -1]), 0
    elif wall_name == 'left':
        # x = 0
        return np.array([-1, 0, 0]), 0
    elif wall_name == 'right':
        # x = width
        return np.array([1, 0, 0]), -w
    elif wall_name == 'floor':
        # y = 0
        return np.array([0, -1, 0]), 0
    elif wall_name == 'ceiling':
        # y = height
        return np.array([0, 1, 0]), -h
    else:
        raise ValueError(f"Unknown wall: {wall_name}")


def ray_plane_intersection(ray: Ray3D, normal: np.ndarray, d: float) -> Optional[np.ndarray]:
    """
    Berechne Schnittpunkt von Strahl mit Ebene.
    Ebene: n·p + d = 0
    Strahl: p = origin + t * direction
    
    Returns:
        3D-Punkt oder None wenn parallel
    """
    denom = np.dot(normal, ray.direction)
    
    if abs(denom) < 1e-6:
        return None  # Parallel
    
    t = -(np.dot(normal, ray.origin) + d) / denom
    
    if t < 0:
        return None  # Hinter Kamera
    
    return ray.origin + t * ray.direction


def project_shadow_to_3d(
    normalized_x: float,
    normalized_y: float,
    wall_name: str,
    camera_position: Dict,
    camera_rotation: Dict,
    fov_y: float,
    room: Dict,
    aspect_ratio: float = 16/9
) -> Optional[np.ndarray]:
    """
    Projiziere Schatten-Punkt auf 3D-Punkt auf der angegebenen Wand.
    """
    ray = unproject_2d_to_ray(
        normalized_x, normalized_y,
        camera_position, camera_rotation,
        fov_y, aspect_ratio
    )
    
    normal, d = get_wall_plane(wall_name, room)
    
    return ray_plane_intersection(ray, normal, d)


# =============================================================================
# LIGHT DIRECTION ESTIMATION
# =============================================================================

def estimate_light_direction_for_pair(
    object_ray: Ray3D,
    shadow_point_3d: np.ndarray,
    initial_guess_t: float = 2.0
) -> Tuple[np.ndarray, float]:
    """
    Für ein einzelnes Punkt-Paar: Schätze Lichtrichtung.
    
    Da das Objekt irgendwo auf dem Strahl liegt, parametrisieren wir:
    object_point = ray.origin + t * ray.direction
    light_direction = shadow_point - object_point (normalisiert)
    
    Returns:
        (light_direction, t) - wobei t die Distanz auf dem Strahl ist
    """
    # Für ein einzelnes Paar ist t nicht eindeutig bestimmbar
    # Wir nehmen einen initialen Schätzwert
    object_point = object_ray.origin + initial_guess_t * object_ray.direction
    light_vec = shadow_point_3d - object_point
    
    if np.linalg.norm(light_vec) < 1e-6:
        return np.array([0, -1, 0]), initial_guess_t  # Default: Sonne von oben
    
    return light_vec / np.linalg.norm(light_vec), initial_guess_t


def estimate_light_direction_multi(
    object_rays: List[Ray3D],
    shadow_points_3d: List[np.ndarray]
) -> Tuple[np.ndarray, List[float], float]:
    """
    Schätze eine gemeinsame Lichtrichtung für mehrere Punkt-Paare.
    
    Optimierungsproblem:
    - Finde L (normalisiert) und t1, t2, t3, ...
    - So dass für alle i: (shadow_i - (ray_i.origin + t_i * ray_i.direction)) parallel zu L
    
    Returns:
        (light_direction, object_distances, residual_error)
    """
    n = len(object_rays)
    
    if n < 2:
        raise ValueError("Mindestens 2 Punkt-Paare erforderlich")
    
    # Initiale Schätzung: Mittlere Richtung aus einzelnen Paaren
    initial_directions = []
    initial_ts = []
    
    for ray, shadow in zip(object_rays, shadow_points_3d):
        # Finde t so dass der Abstand zum Schatten plausibel ist
        # Schätzung: Objekt liegt etwa auf halber Strecke zur Wand
        t_estimate = np.linalg.norm(shadow - ray.origin) * 0.5
        t_estimate = max(0.3, min(t_estimate, 10.0))  # Clamp
        initial_ts.append(t_estimate)
        
        obj_point = ray.origin + t_estimate * ray.direction
        light_vec = shadow - obj_point
        if np.linalg.norm(light_vec) > 1e-6:
            initial_directions.append(light_vec / np.linalg.norm(light_vec))
    
    if not initial_directions:
        return np.array([0, -1, 0]), initial_ts, float('inf')
    
    # Mittlere initiale Richtung
    mean_dir = np.mean(initial_directions, axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)
    
    # Konvertiere zu Kugelkoordinaten für Optimierung
    # L = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    # Aber einfacher: Optimiere direkt L und normalisiere
    
    def objective(params):
        """
        params = [Lx, Ly, Lz, t1, t2, ..., tn]
        """
        L = np.array(params[:3])
        L_norm = np.linalg.norm(L)
        if L_norm < 1e-6:
            return 1e10
        L = L / L_norm
        
        ts = params[3:]
        
        total_error = 0.0
        
        for i, (ray, shadow) in enumerate(zip(object_rays, shadow_points_3d)):
            t = ts[i]
            if t < 0.01:  # Objekt muss vor Kamera sein
                return 1e10
            
            obj_point = ray.origin + t * ray.direction
            actual_vec = shadow - obj_point
            actual_norm = np.linalg.norm(actual_vec)
            
            if actual_norm < 1e-6:
                continue
            
            actual_dir = actual_vec / actual_norm
            
            # Fehler = 1 - cos(angle) = 1 - dot(L, actual_dir)
            # Oder: Kreuzprodukt-Norm (= sin(angle))
            cross = np.cross(L, actual_dir)
            error = np.linalg.norm(cross)  # sin(angle)
            total_error += error ** 2
        
        # Regularisierung: ts sollten positiv und nicht zu groß sein
        for t in ts:
            if t < 0:
                total_error += 1000 * t**2
            if t > 20:
                total_error += 0.01 * (t - 20)**2
        
        return total_error
    
    # Initiale Parameter
    x0 = list(mean_dir) + initial_ts
    
    # Optimierung
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=[
            (-1, 1), (-1, 1), (-1, 1),  # L-Komponenten
            *[(0.01, 20.0) for _ in range(n)]  # t-Werte
        ],
        options={'maxiter': 200}
    )
    
    # Ergebnis extrahieren
    L = np.array(result.x[:3])
    L = L / np.linalg.norm(L)
    ts = list(result.x[3:])
    
    return L, ts, result.fun


def compute_direction_error(
    light_direction: np.ndarray,
    object_rays: List[Ray3D],
    shadow_points_3d: List[np.ndarray],
    object_ts: List[float]
) -> List[float]:
    """
    Berechne Winkel-Fehler (in Grad) für jedes Punkt-Paar.
    """
    errors = []
    
    for ray, shadow, t in zip(object_rays, shadow_points_3d, object_ts):
        obj_point = ray.origin + t * ray.direction
        actual_vec = shadow - obj_point
        actual_norm = np.linalg.norm(actual_vec)
        
        if actual_norm < 1e-6:
            errors.append(0.0)
            continue
        
        actual_dir = actual_vec / actual_norm
        
        # Winkel zwischen L und actual_dir
        dot = np.clip(np.dot(light_direction, actual_dir), -1, 1)
        angle_rad = np.arccos(dot)
        angle_deg = np.rad2deg(angle_rad)
        
        errors.append(angle_deg)
    
    return errors


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_object(
    pairs: List[Dict],
    camera_position: Dict,
    camera_rotation: Dict,
    fov_y: float,
    room: Dict,
    aspect_ratio: float = 16/9
) -> ValidationResult:
    """
    Validiere ein einzelnes Objekt (Intra-Objekt-Konsistenz).
    
    Args:
        pairs: Liste von {objectPoint: {normalizedX, normalizedY}, 
                         shadowPoint: {normalizedX, normalizedY, wall}}
        camera_position: {x, y, z}
        camera_rotation: {x, y, z}
        fov_y: FOV in Grad
        room: {width, depth, height}
    
    Returns:
        ValidationResult
    """
    if len(pairs) < 3:
        return ValidationResult(
            success=False,
            status='error',
            consistency_score=0,
            light_direction=None,
            average_error_deg=0,
            max_error_deg=0,
            message=f"Mindestens 3 Punkt-Paare erforderlich, nur {len(pairs)} vorhanden",
            details={}
        )
    
    # 1. Konvertiere alle Punkte
    object_rays = []
    shadow_points_3d = []
    
    for i, pair in enumerate(pairs):
        obj = pair['objectPoint']
        shadow = pair['shadowPoint']
        
        # Objekt-Strahl
        ray = unproject_2d_to_ray(
            obj['normalizedX'], obj['normalizedY'],
            camera_position, camera_rotation,
            fov_y, aspect_ratio
        )
        object_rays.append(ray)
        
        # Schatten-3D-Punkt
        shadow_3d = project_shadow_to_3d(
            shadow['normalizedX'], shadow['normalizedY'],
            shadow['wall'],
            camera_position, camera_rotation,
            fov_y, room, aspect_ratio
        )
        
        if shadow_3d is None:
            return ValidationResult(
                success=False,
                status='error',
                consistency_score=0,
                light_direction=None,
                average_error_deg=0,
                max_error_deg=0,
                message=f"Schatten-Punkt {i+1} konnte nicht auf Wand '{shadow['wall']}' projiziert werden",
                details={'failed_pair_index': i}
            )
        
        shadow_points_3d.append(shadow_3d)
    
    # 2. Schätze gemeinsame Lichtrichtung
    try:
        light_dir, object_ts, residual = estimate_light_direction_multi(
            object_rays, shadow_points_3d
        )
    except Exception as e:
        return ValidationResult(
            success=False,
            status='error',
            consistency_score=0,
            light_direction=None,
            average_error_deg=0,
            max_error_deg=0,
            message=f"Fehler bei Lichtrichtungs-Schätzung: {str(e)}",
            details={'exception': str(e)}
        )
    
    # 3. Berechne Fehler pro Punkt
    errors_deg = compute_direction_error(
        light_dir, object_rays, shadow_points_3d, object_ts
    )
    
    avg_error = np.mean(errors_deg)
    max_error = np.max(errors_deg)
    
    # 4. Konsistenz-Score berechnen
    # Score = 100 wenn avg_error = 0, 0 wenn avg_error >= 45°
    consistency_score = max(0, 100 - (avg_error / 45) * 100)
    
    # 5. Status bestimmen
    if avg_error < 5:
        status = 'valid'
        message = f"Ausgezeichnete Konsistenz (Ø {avg_error:.1f}°)"
    elif avg_error < 10:
        status = 'valid'
        message = f"Gute Konsistenz (Ø {avg_error:.1f}°)"
    elif avg_error < 20:
        status = 'warning'
        message = f"Akzeptable Konsistenz (Ø {avg_error:.1f}°), Überprüfung empfohlen"
    else:
        status = 'error'
        message = f"Schlechte Konsistenz (Ø {avg_error:.1f}°), Punkte überprüfen!"
    
    # Lichtrichtung in Azimut/Elevation konvertieren
    azimuth = np.rad2deg(np.arctan2(light_dir[0], light_dir[2]))  # X/Z
    elevation = np.rad2deg(np.arcsin(np.clip(-light_dir[1], -1, 1)))  # -Y (Sonne von oben = positive elevation)
    
    return ValidationResult(
        success=True,
        status=status,
        consistency_score=consistency_score,
        light_direction=Point3D.from_array(light_dir),
        average_error_deg=avg_error,
        max_error_deg=max_error,
        message=message,
        details={
            'errors_per_point_deg': errors_deg,
            'object_distances': object_ts,
            'residual': residual,
            'light_azimuth_deg': azimuth,
            'light_elevation_deg': elevation,
            'shadow_points_3d': [
                {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
                for p in shadow_points_3d
            ]
        }
    )


def validate_inter_object(
    objects: List[Dict],
    camera_position: Dict,
    camera_rotation: Dict,
    fov_y: float,
    room: Dict,
    aspect_ratio: float = 16/9
) -> Dict:
    """
    Validiere Inter-Objekt-Konsistenz (alle Objekte in einem Screenshot).
    
    Prüft ob alle Objekte auf dieselbe Lichtquelle zeigen.
    """
    if len(objects) < 2:
        return {
            'success': False,
            'status': 'error',
            'message': 'Mindestens 2 Objekte erforderlich',
            'inter_object_score': 0
        }
    
    # Validiere jedes Objekt einzeln und sammle Lichtrichtungen
    object_results = []
    light_directions = []
    
    for obj in objects:
        if len(obj.get('pairs', [])) < 3:
            continue
        
        result = validate_object(
            obj['pairs'],
            camera_position, camera_rotation,
            fov_y, room, aspect_ratio
        )
        
        object_results.append({
            'object_id': obj.get('id', ''),
            'object_name': obj.get('name', ''),
            'result': result
        })
        
        if result.success and result.light_direction:
            light_directions.append(result.light_direction.to_array())
    
    if len(light_directions) < 2:
        return {
            'success': False,
            'status': 'error',
            'message': 'Nicht genug Objekte mit gültiger Lichtrichtung',
            'inter_object_score': 0,
            'object_results': object_results
        }
    
    # Berechne mittlere Lichtrichtung
    mean_light = np.mean(light_directions, axis=0)
    mean_light = mean_light / np.linalg.norm(mean_light)
    
    # Berechne Abweichung jedes Objekts vom Mittel
    deviations = []
    for ld in light_directions:
        dot = np.clip(np.dot(mean_light, ld), -1, 1)
        angle = np.rad2deg(np.arccos(dot))
        deviations.append(angle)
    
    avg_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    
    # Inter-Object Score
    inter_object_score = max(0, 100 - (avg_deviation / 30) * 100)
    
    # Status
    if avg_deviation < 5:
        status = 'valid'
        message = f"Alle Objekte zeigen konsistent in dieselbe Richtung (Ø {avg_deviation:.1f}°)"
    elif avg_deviation < 15:
        status = 'warning'
        message = f"Leichte Abweichungen zwischen Objekten (Ø {avg_deviation:.1f}°)"
    else:
        status = 'error'
        message = f"Signifikante Abweichungen zwischen Objekten (Ø {avg_deviation:.1f}°)!"
    
    # Azimut/Elevation der gemittelten Richtung
    azimuth = np.rad2deg(np.arctan2(mean_light[0], mean_light[2]))
    elevation = np.rad2deg(np.arcsin(np.clip(-mean_light[1], -1, 1)))
    
    return {
        'success': True,
        'status': status,
        'message': message,
        'inter_object_score': inter_object_score,
        'average_deviation_deg': avg_deviation,
        'max_deviation_deg': max_deviation,
        'mean_light_direction': {
            'x': float(mean_light[0]),
            'y': float(mean_light[1]),
            'z': float(mean_light[2])
        },
        'mean_light_azimuth_deg': azimuth,
        'mean_light_elevation_deg': elevation,
        'deviations_per_object_deg': deviations,
        'object_results': [
            {
                'object_id': r['object_id'],
                'object_name': r['object_name'],
                'status': r['result'].status,
                'consistency_score': r['result'].consistency_score,
                'light_direction': {
                    'x': r['result'].light_direction.x,
                    'y': r['result'].light_direction.y,
                    'z': r['result'].light_direction.z
                } if r['result'].light_direction else None,
                'average_error_deg': r['result'].average_error_deg,
                'message': r['result'].message
            }
            for r in object_results
        ]
    }


def validate_screenshot(
    screenshot_data: Dict,
    calibration: Dict
) -> Dict:
    """
    Validiere einen kompletten Screenshot.
    """
    screenshot_id = screenshot_data.get('screenshotId', '')
    objects = screenshot_data.get('objects', [])
    
    # Finde passende Kalibrierung
    screenshot_calib = None
    for sc in calibration.get('screenshots', []):
        if sc.get('screenshotId') == screenshot_id:
            screenshot_calib = sc
            break
    
    if not screenshot_calib:
        return {
            'success': False,
            'status': 'error',
            'screenshot_id': screenshot_id,
            'message': 'Keine Kalibrierung für Screenshot gefunden'
        }
    
    room = calibration.get('room', {})
    camera_pos = calibration.get('camera', {}).get('position', {})
    camera_rot = screenshot_calib.get('cameraRotation', {})
    fov_y = calibration.get('camera', {}).get('fovY', 60)
    
    # Validiere Inter-Objekt
    result = validate_inter_object(
        objects,
        camera_pos, camera_rot,
        fov_y, room
    )
    
    result['screenshot_id'] = screenshot_id
    
    return result


def validate_all(session_data: Dict) -> Dict:
    """
    Globale Validierung aller Screenshots.
    """
    calibration = session_data.get('calibration')
    shadows = session_data.get('shadows', [])
    
    if not calibration:
        return {
            'success': False,
            'status': 'error',
            'message': 'Keine Kalibrierungsdaten vorhanden',
            'global_score': 0
        }
    
    if not shadows:
        return {
            'success': False,
            'status': 'error',
            'message': 'Keine Schatten-Daten vorhanden',
            'global_score': 0
        }
    
    # Validiere jeden Screenshot
    screenshot_results = []
    all_light_directions = []
    
    for shadow_data in shadows:
        result = validate_screenshot(shadow_data, calibration)
        screenshot_results.append(result)
        
        if result.get('success') and result.get('mean_light_direction'):
            ld = result['mean_light_direction']
            all_light_directions.append(np.array([ld['x'], ld['y'], ld['z']]))
    
    # Berechne globale Konsistenz
    # Hinweis: Bei unterschiedlichen Zeitstempeln sollten die Richtungen
    # sich systematisch ändern (Sonnenbewegung)!
    # Fürs Erste: Prüfe nur die Varianz
    
    if len(all_light_directions) >= 2:
        # Berechne paarweise Winkel
        angles = []
        for i in range(len(all_light_directions)):
            for j in range(i+1, len(all_light_directions)):
                dot = np.clip(np.dot(all_light_directions[i], all_light_directions[j]), -1, 1)
                angles.append(np.rad2deg(np.arccos(dot)))
        
        avg_angle_diff = np.mean(angles) if angles else 0
        cross_screenshot_info = {
            'average_direction_difference_deg': avg_angle_diff,
            'note': 'Unterschiedliche Richtungen können durch Zeitversatz (Sonnenbewegung) erklärt werden'
        }
    else:
        cross_screenshot_info = {}
    
    # Globaler Score (gewichteter Durchschnitt)
    scores = [r.get('inter_object_score', 0) for r in screenshot_results if r.get('success')]
    global_score = np.mean(scores) if scores else 0
    
    # Status
    if global_score >= 80:
        status = 'valid'
        message = 'Alle Daten sind konsistent'
    elif global_score >= 50:
        status = 'warning'
        message = 'Einige Inkonsistenzen gefunden'
    else:
        status = 'error'
        message = 'Signifikante Probleme mit den Daten'
    
    return {
        'success': True,
        'status': status,
        'message': message,
        'global_score': global_score,
        'screenshot_results': screenshot_results,
        'cross_screenshot_consistency': cross_screenshot_info,
        'summary': {
            'total_screenshots': len(shadows),
            'valid_screenshots': sum(1 for r in screenshot_results if r.get('status') == 'valid'),
            'warning_screenshots': sum(1 for r in screenshot_results if r.get('status') == 'warning'),
            'error_screenshots': sum(1 for r in screenshot_results if r.get('status') == 'error')
        }
    }
