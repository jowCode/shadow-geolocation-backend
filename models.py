"""
Pydantic Models für Shadow Geolocation Backend

VERSION 2.0:
- FOV statt master_focal_length
- Normalisierte Koordinaten für Schatten-Daten
- Klare Trennung zwischen mathematischen und Display-Parametern
"""

from pydantic import BaseModel
from typing import List, Optional, Dict


# ============================================================================
# SESSION MODELS
# ============================================================================

class SessionCreate(BaseModel):
    project_name: str
    camera_type: str


class SessionResponse(BaseModel):
    session_id: str
    project_name: str


# ============================================================================
# SOLVE MODELS (für spätere Geolocation-Berechnung)
# ============================================================================

class ShadowPair(BaseModel):
    method: str
    object_point: List[float]
    shadow_point: List[float]
    confidence: float


class ShadowMeasurement(BaseModel):
    time_offset_seconds: int
    pairs: List[ShadowPair]


class Constraints(BaseModel):
    hemisphere: str
    lat_range: List[float]
    lon_range: List[float]
    date_range: Optional[dict] = None
    time_range: Optional[dict] = None


class SolveRequest(BaseModel):
    session_id: str
    shadow_data: List[ShadowMeasurement]
    constraints: Constraints
    settings: dict


class Solution(BaseModel):
    lat: float
    lon: float
    date: str
    time: str
    error: float
    uncertainty: float
    location_name: str


# ============================================================================
# BUNDLE ADJUSTMENT MODELS (v2.0)
# ============================================================================

class DisplayParams(BaseModel):
    """
    UI-Parameter für die Screenshot-Darstellung.
    
    WICHTIG: Diese haben KEINE mathematische Bedeutung für die 3D-Rekonstruktion!
    Sie dienen nur der visuellen Ausrichtung im Frontend.
    
    Der backgroundOffset ist jedoch ein ERROR-SIGNAL:
    Wenn offset != (50, 50), bedeutet das, dass Raum/Kamera/FOV nicht perfekt kalibriert sind.
    """
    background_scale: float = 50.0      # CSS zoom (%)
    background_rotation: float = 0.0    # CSS rotation (°)
    background_offset_x: float = 50.0   # CSS position X (%)
    background_offset_y: float = 50.0   # CSS position Y (%)


class EulerRotation(BaseModel):
    """Euler-Rotation mit expliziter Order"""
    x: float = 0.0  # Pitch (°)
    y: float = 0.0  # Yaw (°)
    z: float = 0.0  # Roll (°)
    order: str = "YXZ"


class ScreenshotDimensions(BaseModel):
    """Original-Dimensionen des Screenshots"""
    width: int
    height: int


class CalibrationScreenshotData(BaseModel):
    """
    Kalibrierungs-Daten für einen Screenshot (v2.0)
    
    Änderungen:
    - camera_rotation statt room_rotation (semantisch klarer)
    - display Parameter getrennt von mathematischen Parametern
    - Screenshot-Dimensionen für Normalisierung
    """
    id: str
    
    # NEU: Screenshot-Dimensionen (für normalisierte Koordinaten)
    screenshot_dimensions: Optional[ScreenshotDimensions] = None
    
    # Kamera-Blickrichtung (Pan/Tilt) für diesen Screenshot
    camera_rotation: Optional[EulerRotation] = None
    
    # Display-Parameter (NUR für UI!)
    display: Optional[DisplayParams] = None
    
    # Legacy-Felder (für Abwärtskompatibilität)
    camera_position: Optional[dict] = None  # Wird ignoriert, da global
    room_rotation: Optional[dict] = None    # Migration zu camera_rotation
    background_rotation: Optional[float] = None
    background_scale: Optional[float] = None
    background_offset_x: Optional[float] = None
    background_offset_y: Optional[float] = None
    
    completed: bool = False
    
    def get_camera_rotation(self) -> Dict[str, float]:
        """Hole Kamera-Rotation (mit Legacy-Fallback)"""
        if self.camera_rotation:
            return {
                'x': self.camera_rotation.x,
                'y': self.camera_rotation.y,
                'z': self.camera_rotation.z
            }
        elif self.room_rotation:
            return self.room_rotation
        else:
            return {'x': 0, 'y': 0, 'z': 0}
    
    def get_display_params(self) -> Dict[str, float]:
        """Hole Display-Parameter (mit Legacy-Fallback)"""
        if self.display:
            return {
                'background_scale': self.display.background_scale,
                'background_rotation': self.display.background_rotation,
                'background_offset_x': self.display.background_offset_x,
                'background_offset_y': self.display.background_offset_y
            }
        else:
            return {
                'background_scale': self.background_scale or 50.0,
                'background_rotation': self.background_rotation or 0.0,
                'background_offset_x': self.background_offset_x or 50.0,
                'background_offset_y': self.background_offset_y or 50.0
            }


class GlobalCameraParams(BaseModel):
    """Globale Kamera-Parameter (für alle Screenshots gleich)"""
    position: Dict[str, float]  # {x, y, z} in Metern
    fov_y: float = 60.0         # Vertikales FOV in Grad


class BundleAdjustmentRequest(BaseModel):
    """
    Request für Bundle Adjustment (v2.0)
    
    Änderungen:
    - camera: GlobalCameraParams mit position + fov_y
    - global_display_zoom: Ersetzt master_focal_length (nur UI!)
    - Legacy-Felder für Abwärtskompatibilität
    """
    session_id: str
    
    # Raum-Dimensionen
    room: Dict[str, float]  # {width, depth, height}
    
    # NEU: Globale Kamera-Parameter (v2.0)
    camera: Optional[GlobalCameraParams] = None
    
    # NEU: Globaler Display-Zoom (nur UI, keine mathematische Bedeutung)
    global_display_zoom: Optional[float] = None
    
    # Screenshots
    screenshots: List[CalibrationScreenshotData]
    
    # Optimierungs-Gewichte
    weights: Optional[Dict[str, float]] = None  # {room_confidence, position_confidence}
    
    # Legacy-Felder (für Abwärtskompatibilität)
    global_camera_position: Optional[Dict[str, float]] = None
    master_focal_length: Optional[float] = None
    global_fov_y: Optional[float] = None
    
    def get_camera_position(self) -> Dict[str, float]:
        """Hole Kamera-Position (mit Legacy-Fallback)"""
        if self.camera and self.camera.position:
            return self.camera.position
        elif self.global_camera_position:
            return self.global_camera_position
        else:
            return {'x': 2.0, 'y': 1.5, 'z': 0.5}
    
    def get_fov_y(self) -> float:
        """Hole FOV (mit Legacy-Fallback)"""
        if self.camera and self.camera.fov_y:
            return self.camera.fov_y
        elif self.global_fov_y:
            return self.global_fov_y
        else:
            # Legacy: master_focal_length als Fallback (aber nicht korrekt!)
            return 60.0  # Default FOV
    
    def get_weights(self) -> Dict[str, float]:
        """Hole Gewichte mit Defaults"""
        return self.weights or {
            'room_confidence': 0.5,
            'position_confidence': 0.5
        }


class BundleAdjustmentResult(BaseModel):
    """Ergebnis des Bundle Adjustments"""
    optimized_room: Dict[str, float]
    optimized_camera: Optional[Dict[str, float]]
    optimized_fov: Optional[float] = None  # NEU: Optimiertes FOV
    initial_error: float
    final_error: float
    improvement_percent: float
    iterations: int
    success: bool
    message: str
    
    # Positions-Varianz (für Diagnostik)
    positions_variance_before: Optional[float] = None
    positions_variance_after: Optional[float] = None
    variance_reduction_percent: Optional[float] = None


# ============================================================================
# SHADOW DATA MODELS (v2.0) - Normalisierte Koordinaten
# ============================================================================

class NormalizedPoint(BaseModel):
    """Normalisierter 2D-Punkt (0-1)"""
    normalized_x: float
    normalized_y: float


class ShadowPointData(BaseModel):
    """Schatten-Punkt mit Wand-Information"""
    normalized_x: float
    normalized_y: float
    wall: str  # 'back', 'left', 'right', 'front', 'floor'
    
    # Optional: 3D-Position auf der Wand (für Debugging)
    world_3d: Optional[Dict[str, float]] = None


class ShadowPairData(BaseModel):
    """Ein Punkt-Paar (Objekt → Schatten)"""
    object_point: NormalizedPoint
    shadow_point: ShadowPointData


class ShadowObjectData(BaseModel):
    """Ein Objekt mit seinen Schatten-Paaren"""
    id: str
    name: str
    pairs: List[ShadowPairData]


class ScreenshotShadowData(BaseModel):
    """Schatten-Daten für einen Screenshot"""
    screenshot_id: str
    timestamp: str
    screenshot_dimensions: Optional[ScreenshotDimensions] = None
    objects: List[ShadowObjectData]


class ShadowDataRequest(BaseModel):
    """Vollständige Schatten-Daten (v2.0)"""
    version: str = "2.0"
    screenshots: List[ScreenshotShadowData]