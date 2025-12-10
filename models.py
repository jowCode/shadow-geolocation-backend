"""
Session Models für Shadow Geolocation Backend

VERSION 3.0:
- Zentrales SessionData Model (alles in einer session.json)
- Kein Legacy v1/v2 Support mehr
- Klare Struktur für alle Stages
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# ============================================================================
# BASIC TYPES
# ============================================================================

class Point3D(BaseModel):
    """3D-Punkt in Metern"""
    x: float
    y: float
    z: float


class NormalizedPoint2D(BaseModel):
    """Normalisierter 2D-Punkt (0-1)"""
    normalizedX: float
    normalizedY: float


class Dimensions(BaseModel):
    """Bild-Dimensionen in Pixeln"""
    width: int
    height: int


class EulerRotation(BaseModel):
    """Euler-Rotation in Grad"""
    x: float = 0.0  # Pitch
    y: float = 0.0  # Yaw
    z: float = 0.0  # Roll


class RoomDimensions(BaseModel):
    """Raum-Dimensionen in Metern"""
    width: float = Field(gt=0, description="X-Achse")
    depth: float = Field(gt=0, description="Z-Achse")
    height: float = Field(gt=0, description="Y-Achse")


# ============================================================================
# META (Stage 1)
# ============================================================================

class MetaData(BaseModel):
    """Projekt-Metadaten"""
    projectName: str
    cameraType: Literal["static"] = "static"
    createdAt: str  # ISO DateTime
    lastModified: str  # ISO DateTime


# ============================================================================
# SCREENSHOTS (Stage 1)
# ============================================================================

class ScreenshotData(BaseModel):
    """Ein Screenshot mit Metadaten"""
    id: str
    filename: str
    timestamp: str  # "t0" | "t0+30" | "t0+60" etc.
    isReferencePoint: bool = False  # Ist dies t0?
    dimensions: Optional[Dimensions] = None


# ============================================================================
# CALIBRATION (Stage 3)
# ============================================================================

class CameraParams(BaseModel):
    """Globale Kamera-Parameter"""
    position: Point3D
    fovY: float = Field(gt=0, lt=180, description="Vertikales FOV in Grad")


class DisplayParams(BaseModel):
    """UI-Parameter für Screenshot-Darstellung (keine mathematische Bedeutung!)"""
    backgroundScale: float = 50.0
    backgroundRotation: float = 0.0
    backgroundOffsetX: float = 50.0
    backgroundOffsetY: float = 50.0


class ScreenshotCalibration(BaseModel):
    """Kalibrierung für einen einzelnen Screenshot"""
    screenshotId: str
    cameraRotation: EulerRotation
    display: DisplayParams
    completed: bool = False


class CalibrationData(BaseModel):
    """Vollständige Kalibrierungsdaten"""
    room: RoomDimensions
    camera: CameraParams
    globalDisplayZoom: float = 50.0  # UI-only
    screenshots: List[ScreenshotCalibration] = []


# ============================================================================
# SHADOWS (Stage 5)
# ============================================================================

WallName = Literal["back", "left", "right", "front", "floor"]


class ShadowPoint(BaseModel):
    """Schatten-Punkt mit Wand-Information"""
    normalizedX: float
    normalizedY: float
    wall: WallName
    world3D: Optional[Point3D] = None  # Optional, für Debug


class ShadowPair(BaseModel):
    """Ein Punkt-Paar (Objekt → Schatten)"""
    objectPoint: NormalizedPoint2D
    shadowPoint: ShadowPoint


class ShadowObject(BaseModel):
    """Ein Objekt mit seinen Schatten-Paaren"""
    id: str
    name: str
    pairs: List[ShadowPair] = []


class ScreenshotShadows(BaseModel):
    """Schatten-Daten für einen Screenshot"""
    screenshotId: str
    objects: List[ShadowObject] = []


# ============================================================================
# VALIDATION (Stage 6)
# ============================================================================

ValidationStatus = Literal["pending", "valid", "warning", "error"]


class ObjectValidation(BaseModel):
    """Validierung für ein Objekt"""
    objectId: str
    status: ValidationStatus = "pending"
    consistencyScore: Optional[float] = None
    message: Optional[str] = None


class ScreenshotValidation(BaseModel):
    """Validierung für einen Screenshot"""
    screenshotId: str
    status: ValidationStatus = "pending"
    intraObjectScore: Optional[float] = None
    interObjectScore: Optional[float] = None
    objects: List[ObjectValidation] = []


class ValidationData(BaseModel):
    """Validierungsdaten"""
    lastRun: Optional[str] = None  # ISO DateTime
    globalStatus: ValidationStatus = "pending"
    globalScore: Optional[float] = None
    screenshots: List[ScreenshotValidation] = []


# ============================================================================
# CENTRAL SESSION MODEL
# ============================================================================

class SessionData(BaseModel):
    """
    Zentrales Session-Model (v3.0)
    
    Enthält ALLE Daten einer Session in einer Struktur.
    Wird als session.json gespeichert.
    """
    version: str = "3.0"
    sessionId: str
    
    # Stage 1: Meta + Screenshots
    meta: MetaData
    screenshots: List[ScreenshotData] = []
    
    # Stage 3: Calibration (null wenn noch nicht begonnen)
    calibration: Optional[CalibrationData] = None
    
    # Stage 5: Shadows (null wenn noch nicht begonnen)
    shadows: Optional[List[ScreenshotShadows]] = None
    
    # Stage 6: Validation (optional)
    validation: Optional[ValidationData] = None


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request zum Erstellen einer neuen Session"""
    projectName: str
    cameraType: Literal["static"] = "static"
    screenshots: List[ScreenshotData]


class CreateSessionResponse(BaseModel):
    """Response nach Session-Erstellung"""
    sessionId: str
    projectName: str


class UploadScreenshotResponse(BaseModel):
    """Response nach Screenshot-Upload"""
    filename: str
    screenshotId: str
    url: str


# ============================================================================
# BUNDLE ADJUSTMENT MODELS (für WebSocket)
# ============================================================================

class BundleAdjustmentRequest(BaseModel):
    """Request für Bundle Adjustment"""
    sessionId: str
    calibration: CalibrationData
    weights: Optional[dict] = None


class BundleAdjustmentResult(BaseModel):
    """Ergebnis des Bundle Adjustments"""
    optimizedRoom: RoomDimensions
    optimizedCamera: Optional[Point3D] = None
    initialError: float
    finalError: float
    improvementPercent: float
    iterations: int
    success: bool
    message: str = ""
