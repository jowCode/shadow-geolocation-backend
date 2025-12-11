"""
Validation API Endpoints für Shadow Geolocation

Ergänzt main.py um Validierungs-Funktionen.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

from .validation import (
    validate_object,
    validate_inter_object,
    validate_screenshot,
    validate_all
)

router = APIRouter(prefix="/api/sessions", tags=["validation"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ValidateObjectRequest(BaseModel):
    """Request für Einzelobjekt-Validierung"""
    screenshotId: str
    objectId: str


class ValidateInterObjectRequest(BaseModel):
    """Request für Inter-Objekt-Validierung"""
    screenshotId: str


class ValidationResponse(BaseModel):
    """Allgemeine Validierungs-Response"""
    success: bool
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_session_data(session_id: str, load_session_func) -> Dict:
    """Lädt Session-Daten"""
    session = load_session_func(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return session.model_dump()


def find_screenshot_data(session_data: Dict, screenshot_id: str) -> Optional[Dict]:
    """Findet Schatten-Daten für einen Screenshot"""
    shadows = session_data.get('shadows', [])
    for s in shadows:
        if s.get('screenshotId') == screenshot_id:
            return s
    return None


def find_object_data(screenshot_data: Dict, object_id: str) -> Optional[Dict]:
    """Findet ein Objekt in den Screenshot-Daten"""
    for obj in screenshot_data.get('objects', []):
        if obj.get('id') == object_id:
            return obj
    return None


def find_screenshot_calibration(session_data: Dict, screenshot_id: str) -> Optional[Dict]:
    """Findet die Kalibrierung für einen Screenshot"""
    calibration = session_data.get('calibration')
    if not calibration:
        return None
    
    for sc in calibration.get('screenshots', []):
        if sc.get('screenshotId') == screenshot_id:
            return sc
    return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

def create_validation_routes(load_session_func):
    """
    Factory-Funktion um Routes mit Zugriff auf load_session zu erstellen.
    
    Wird in main.py aufgerufen mit der load_session Funktion.
    """
    
    @router.post("/{session_id}/validate/object", response_model=ValidationResponse)
    async def validate_single_object(session_id: str, request: ValidateObjectRequest):
        """
        Validiert ein einzelnes Objekt (Intra-Objekt-Konsistenz).
        
        Prüft ob die 3 Punkt-Paare des Objekts konsistent sind,
        d.h. ob sie alle auf dieselbe Lichtrichtung zeigen.
        """
        session_data = get_session_data(session_id, load_session_func)
        
        # Finde Screenshot-Daten
        screenshot_data = find_screenshot_data(session_data, request.screenshotId)
        if not screenshot_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Screenshot {request.screenshotId} nicht in Schatten-Daten gefunden"
            )
        
        # Finde Objekt
        obj = find_object_data(screenshot_data, request.objectId)
        if not obj:
            raise HTTPException(
                status_code=404,
                detail=f"Objekt {request.objectId} nicht gefunden"
            )
        
        # Finde Kalibrierung
        screenshot_calib = find_screenshot_calibration(session_data, request.screenshotId)
        if not screenshot_calib:
            raise HTTPException(
                status_code=404,
                detail=f"Keine Kalibrierung für Screenshot {request.screenshotId}"
            )
        
        calibration = session_data.get('calibration', {})
        room = calibration.get('room', {})
        camera_pos = calibration.get('camera', {}).get('position', {})
        camera_rot = screenshot_calib.get('cameraRotation', {})
        fov_y = calibration.get('camera', {}).get('fovY', 60)
        
        # Validierung durchführen
        result = validate_object(
            obj.get('pairs', []),
            camera_pos,
            camera_rot,
            fov_y,
            room
        )
        
        return ValidationResponse(
            success=result.success,
            status=result.status,
            message=result.message,
            data={
                'objectId': request.objectId,
                'screenshotId': request.screenshotId,
                'consistencyScore': result.consistency_score,
                'lightDirection': {
                    'x': result.light_direction.x,
                    'y': result.light_direction.y,
                    'z': result.light_direction.z
                } if result.light_direction else None,
                'averageErrorDeg': result.average_error_deg,
                'maxErrorDeg': result.max_error_deg,
                'details': result.details
            }
        )
    
    
    @router.post("/{session_id}/validate/inter-object", response_model=ValidationResponse)
    async def validate_inter_object_endpoint(session_id: str, request: ValidateInterObjectRequest):
        """
        Validiert Inter-Objekt-Konsistenz für einen Screenshot.
        
        Prüft ob alle Objekte in einem Screenshot auf dieselbe
        Lichtquelle (Sonnenrichtung) zeigen.
        """
        session_data = get_session_data(session_id, load_session_func)
        
        # Finde Screenshot-Daten
        screenshot_data = find_screenshot_data(session_data, request.screenshotId)
        if not screenshot_data:
            raise HTTPException(
                status_code=404,
                detail=f"Screenshot {request.screenshotId} nicht in Schatten-Daten gefunden"
            )
        
        # Finde Kalibrierung
        screenshot_calib = find_screenshot_calibration(session_data, request.screenshotId)
        if not screenshot_calib:
            raise HTTPException(
                status_code=404,
                detail=f"Keine Kalibrierung für Screenshot {request.screenshotId}"
            )
        
        calibration = session_data.get('calibration', {})
        room = calibration.get('room', {})
        camera_pos = calibration.get('camera', {}).get('position', {})
        camera_rot = screenshot_calib.get('cameraRotation', {})
        fov_y = calibration.get('camera', {}).get('fovY', 60)
        
        # Validierung durchführen
        result = validate_inter_object(
            screenshot_data.get('objects', []),
            camera_pos,
            camera_rot,
            fov_y,
            room
        )
        
        return ValidationResponse(
            success=result.get('success', False),
            status=result.get('status', 'error'),
            message=result.get('message', ''),
            data={
                'screenshotId': request.screenshotId,
                'interObjectScore': result.get('inter_object_score', 0),
                'averageDeviationDeg': result.get('average_deviation_deg', 0),
                'maxDeviationDeg': result.get('max_deviation_deg', 0),
                'meanLightDirection': result.get('mean_light_direction'),
                'meanLightAzimuthDeg': result.get('mean_light_azimuth_deg'),
                'meanLightElevationDeg': result.get('mean_light_elevation_deg'),
                'objectResults': result.get('object_results', [])
            }
        )
    
    
    @router.post("/{session_id}/validate/screenshot", response_model=ValidationResponse)
    async def validate_screenshot_endpoint(session_id: str, request: ValidateInterObjectRequest):
        """
        Validiert einen kompletten Screenshot (Alias für inter-object).
        """
        return await validate_inter_object_endpoint(session_id, request)
    
    
    @router.post("/{session_id}/validate/all", response_model=ValidationResponse)
    async def validate_all_endpoint(session_id: str):
        """
        Validiert alle Daten einer Session.
        
        Führt Intra-Objekt, Inter-Objekt und Cross-Screenshot-Validierung durch.
        """
        session_data = get_session_data(session_id, load_session_func)
        
        # Validierung durchführen
        result = validate_all(session_data)
        
        return ValidationResponse(
            success=result.get('success', False),
            status=result.get('status', 'error'),
            message=result.get('message', ''),
            data={
                'globalScore': result.get('global_score', 0),
                'summary': result.get('summary', {}),
                'screenshotResults': result.get('screenshot_results', []),
                'crossScreenshotConsistency': result.get('cross_screenshot_consistency', {})
            }
        )
    
    return router
