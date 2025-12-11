"""
Shadow Geolocation Backend

VERSION 3.0:
- Zentrale session.json (alle Daten in einer Datei)
- Vereinfachte API (4 Endpoints statt 8+)
- Kein Legacy v1/v2 Support
- NEU: Validation Endpoints
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import json
from pathlib import Path
import uuid
from datetime import datetime

# Models
from models import (
    SessionData, CreateSessionRequest, CreateSessionResponse,
    UploadScreenshotResponse, MetaData, CalibrationData,
    RoomDimensions, CameraParams, Point3D, ScreenshotCalibration,
    EulerRotation, DisplayParams, BundleAdjustmentRequest
)

# Solver
from solver.bundle_adjustment import bundle_adjustment_async
from solver.geolocation import calculate_geolocation, get_sun_position

# Validation Solver
from solver.validation import (
    validate_object as validate_object_func,
    validate_inter_object as validate_inter_object_func,
    validate_screenshot as validate_screenshot_func,
    validate_all as validate_all_func
)

# FastAPI App
app = FastAPI(
    title="Shadow Geolocation Backend",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
DATA_DIR = Path("data/uploads")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_session_dir(session_id: str) -> Path:
    return DATA_DIR / session_id


def get_session_file(session_id: str) -> Path:
    return get_session_dir(session_id) / "session.json"


def get_screenshots_dir(session_id: str) -> Path:
    return get_session_dir(session_id) / "screenshots"


def load_session(session_id: str) -> Optional[SessionData]:
    session_file = get_session_file(session_id)
    if not session_file.exists():
        return None

    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SessionData(**data)


def save_session(session_data: SessionData) -> None:
    session_file = get_session_file(session_data.sessionId)

    session_data.meta.lastModified = datetime.utcnow().isoformat() + "Z"

    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Session gespeichert: {session_data.sessionId}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"status": "Shadow Geolocation Backend v3.0 running"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0"}


# ----------------------------------------------------------------------------
# SESSION CRUD
# ----------------------------------------------------------------------------

@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    session_dir = get_session_dir(session_id)
    session_dir.mkdir(exist_ok=True)
    get_screenshots_dir(session_id).mkdir(exist_ok=True)

    session_data = SessionData(
        version="3.0",
        sessionId=session_id,
        meta=MetaData(
            projectName=request.projectName,
            cameraType=request.cameraType,
            createdAt=now,
            lastModified=now
        ),
        screenshots=request.screenshots,
        calibration=None,
        shadows=None,
        validation=None
    )

    save_session(session_data)

    return CreateSessionResponse(
        sessionId=session_id,
        projectName=request.projectName
    )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session_data = load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    return session_data.model_dump()


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, session_data: SessionData):
    if not get_session_dir(session_id).exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    if session_data.sessionId != session_id:
        raise HTTPException(status_code=400, detail="Session-ID stimmt nicht überein")

    save_session(session_data)
    return {"status": "saved", "sessionId": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_dir = get_session_dir(session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    import shutil
    shutil.rmtree(session_dir)

    return {"status": "deleted", "sessionId": session_id}


# ----------------------------------------------------------------------------
# SCREENSHOTS
# ----------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/screenshots", response_model=UploadScreenshotResponse)
async def upload_screenshot(session_id: str, screenshot_id: str, file: UploadFile = File(...)):
    session_dir = get_session_dir(session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    screenshots_dir = get_screenshots_dir(session_id)
    screenshots_dir.mkdir(exist_ok=True)

    filename = f"{screenshot_id}.png"
    file_path = screenshots_dir / filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return UploadScreenshotResponse(
        filename=filename,
        screenshotId=screenshot_id,
        url=f"/api/sessions/{session_id}/screenshots/{filename}"
    )


@app.get("/api/sessions/{session_id}/screenshots/{filename}")
async def get_screenshot(session_id: str, filename: str):
    file_path = get_screenshots_dir(session_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot nicht gefunden")

    return FileResponse(file_path, media_type="image/png")


# ============================================================================
# VALIDATION ENDPOINTS
# ============================================================================

class ValidateObjectRequest(BaseModel):
    screenshotId: str
    objectId: str


class ValidateInterObjectRequest(BaseModel):
    screenshotId: str


class ValidationResponse(BaseModel):
    success: bool
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


def find_screenshot_data(session_data: dict, screenshot_id: str) -> Optional[dict]:
    shadows = session_data.get('shadows', [])
    for s in shadows:
        if s.get('screenshotId') == screenshot_id:
            return s
    return None


def find_object_data(screenshot_data: dict, object_id: str) -> Optional[dict]:
    for obj in screenshot_data.get('objects', []):
        if obj.get('id') == object_id:
            return obj
    return None


def find_screenshot_calibration(session_data: dict, screenshot_id: str) -> Optional[dict]:
    calibration = session_data.get('calibration')
    if not calibration:
        return None
    for sc in calibration.get('screenshots', []):
        if sc.get('screenshotId') == screenshot_id:
            return sc
    return None


@app.post("/api/sessions/{session_id}/validate/object", response_model=ValidationResponse)
async def validate_object(session_id: str, request: ValidateObjectRequest):
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session_data = session.model_dump()
    screenshot_data = find_screenshot_data(session_data, request.screenshotId)

    if not screenshot_data:
        raise HTTPException(status_code=404, detail=f"Screenshot {request.screenshotId} nicht in Schatten-Daten gefunden")

    obj = find_object_data(screenshot_data, request.objectId)
    if not obj:
        raise HTTPException(status_code=404, detail=f"Objekt {request.objectId} nicht gefunden")

    screenshot_calib = find_screenshot_calibration(session_data, request.screenshotId)
    if not screenshot_calib:
        raise HTTPException(status_code=404, detail=f"Keine Kalibrierung für Screenshot {request.screenshotId}")

    calibration = session_data.get('calibration', {})
    room = calibration.get('room', {})
    camera_pos = calibration.get('camera', {}).get('position', {})
    camera_rot = screenshot_calib.get('cameraRotation', {})
    fov_y = calibration.get('camera', {}).get('fovY', 60)

    result = validate_object_func(
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


@app.post("/api/sessions/{session_id}/validate/inter-object", response_model=ValidationResponse)
async def validate_inter_object(session_id: str, request: ValidateInterObjectRequest):
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session_data = session.model_dump()
    screenshot_data = find_screenshot_data(session_data, request.screenshotId)

    if not screenshot_data:
        raise HTTPException(status_code=404, detail=f"Screenshot {request.screenshotId} nicht in Schatten-Daten gefunden")

    screenshot_calib = find_screenshot_calibration(session_data, request.screenshotId)
    if not screenshot_calib:
        raise HTTPException(status_code=404, detail=f"Keine Kalibrierung für Screenshot {request.screenshotId}")

    calibration = session_data.get('calibration', {})
    room = calibration.get('room', {})
    camera_pos = calibration.get('camera', {}).get('position', {})
    camera_rot = screenshot_calib.get('cameraRotation', {})
    fov_y = calibration.get('camera', {}).get('fovY', 60)

    result = validate_inter_object_func(
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


@app.post("/api/sessions/{session_id}/validate/all", response_model=ValidationResponse)
async def validate_all(session_id: str):
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session_data = session.model_dump()
    result = validate_all_func(session_data)

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


# ============================================================================
# BUNDLE ADJUSTMENT (WebSocket)
# ============================================================================

@app.websocket("/ws/bundle-adjustment")
async def websocket_bundle_adjustment(websocket: WebSocket):
    await websocket.accept()

    try:
        raw_data = await websocket.receive_json()

        if 'calibration' not in raw_data:
            await websocket.send_json({
                "type": "error",
                "message": "Keine Kalibrierungsdaten im Request"
            })
            return

        calibration_data = raw_data['calibration']
        weights = raw_data.get('weights', {
            'room_confidence': 0.5,
            'position_confidence': 0.5
        })

        async for update in bundle_adjustment_async(calibration_data, weights):
            await websocket.send_json(update)
            if update['type'] in ['error', 'result']:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server-Fehler: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# ============================================================================
# GEOLOCATION ENDPOINTS (Stage 7)
# ============================================================================

class GeolocationRequest(BaseModel):
    screenshot_id: str
    date: str
    time_utc: str
    hemisphere: str = "north"
    room_orientation: float = 0.0


@app.post("/api/sessions/{session_id}/geolocation")
async def compute_geolocation(session_id: str, request: GeolocationRequest):
    session_path = get_session_file(session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    with open(session_path, 'r') as f:
        session_data = json.load(f)

    result = calculate_geolocation(
        session_data=session_data,
        screenshot_id=request.screenshot_id,
        date_str=request.date,
        time_str=request.time_utc,
        hemisphere=request.hemisphere,
        room_orientation=request.room_orientation
    )

    return result


@app.get("/api/sun-position")
async def get_sun_position_api(
    latitude: float,
    longitude: float,
    date: str,
    time_utc: str
):
    from datetime import datetime, timezone

    try:
        dt = datetime.strptime(f"{date} {time_utc}", "%Y-%m-%d %H:%M")
        dt = dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ungültiges Datum/Uhrzeit: {e}")

    sun = get_sun_position(latitude, longitude, dt)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "datetime_utc": dt.isoformat(),
        "azimuth": round(sun.azimuth, 2),
        "elevation": round(sun.elevation, 2)
    }


# ============================================================================
# LEGACY ENDPOINTS (TEMPORÄR)
# ============================================================================

# … (Unveränderte Legacy-Blöcke)
# Die Legacy Endpoints bleiben unverändert, da die Frage sich nur auf Geolocation & Imports bezog.

@app.post("/api/session/create")
async def legacy_create_session(data: dict):
    print("LEGACY ENDPOINT: /api/session/create")

    request = CreateSessionRequest(
        projectName=data.get('project_name', 'Unnamed'),
        cameraType=data.get('camera_type', 'static'),
        screenshots=[]
    )

    response = await create_session(request)

    return {
        "session_id": response.sessionId,
        "project_name": response.projectName
    }


@app.get("/api/session/{session_id}/calibration")
async def legacy_load_calibration(session_id: str):
    print(f"LEGACY ENDPOINT: load calibration {session_id}")

    session_data = load_session(session_id)
    if not session_data:
        return {"status": "not_found", "data": None}

    if not session_data.calibration:
        return {"status": "not_found", "data": None}

    legacy_data = {
        "version": "2.0",
        "room": session_data.calibration.room.model_dump(),
        "camera": session_data.calibration.camera.model_dump(),
        "globalDisplayZoom": session_data.calibration.globalDisplayZoom,
        "globalCameraPosition": session_data.calibration.camera.position.model_dump(),
        "globalFovY": session_data.calibration.camera.fovY,
        "screenshots": [
            {
                "id": s.screenshotId,
                "cameraRotation": s.cameraRotation.model_dump(),
                "display": s.display.model_dump(),
                "roomRotation": s.cameraRotation.model_dump(),
                "backgroundRotation": s.display.backgroundRotation,
                "backgroundScale": s.display.backgroundScale,
                "backgroundOffsetX": s.display.backgroundOffsetX,
                "backgroundOffsetY": s.display.backgroundOffsetY,
                "completed": s.completed
            }
            for s in session_data.calibration.screenshots
        ]
    }

    return {"status": "found", "data": legacy_data}


@app.post("/api/session/{session_id}/calibration")
async def legacy_save_calibration(session_id: str, data: dict):
    print(f"LEGACY ENDPOINT: save calibration {session_id}")

    session_data = load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    room_data = data.get('room', {})
    camera_pos = data.get('camera', {}).get('position') or data.get('globalCameraPosition', {})
    fov_y = data.get('camera', {}).get('fovY') or data.get('globalFovY', 60)

    session_data.calibration = CalibrationData(
        room=RoomDimensions(**room_data),
        camera=CameraParams(
            position=Point3D(**camera_pos),
            fovY=fov_y
        ),
        globalDisplayZoom=data.get('globalDisplayZoom', data.get('masterFocalLength', 50)),
        screenshots=[
            ScreenshotCalibration(
                screenshotId=s.get('id', ''),
                cameraRotation=EulerRotation(**s.get('cameraRotation', s.get('roomRotation', {}))),
                display=DisplayParams(**s.get('display', {
                    'backgroundScale': s.get('backgroundScale', 50),
                    'backgroundRotation': s.get('backgroundRotation', 0),
                    'backgroundOffsetX': s.get('backgroundOffsetX', 50),
                    'backgroundOffsetY': s.get('backgroundOffsetY', 50)
                })),
                completed=s.get('completed', False)
            )
            for s in data.get('screenshots', [])
        ]
    )

    save_session(session_data)

    return {"status": "saved"}


@app.get("/api/session/{session_id}/organize")
async def legacy_load_organization(session_id: str):
    print(f"LEGACY ENDPOINT: load organization {session_id}")

    session_data = load_session(session_id)
    if not session_data:
        return {"status": "not_found", "data": None}

    legacy_data = {
        "screenshots": [
            {
                "id": s.id,
                "filename": s.filename,
                "timestamp": s.timestamp,
                "useForCalibration": True,
                "timestampType": "reference" if s.isReferencePoint else "offset"
            }
            for s in session_data.screenshots
        ]
    }

    return {"status": "found", "data": legacy_data}


@app.post("/api/session/{session_id}/organize")
async def legacy_save_organization(session_id: str, data: dict):
    print(f"LEGACY ENDPOINT: save organization {session_id}")
    return {"status": "saved"}


@app.post("/api/session/{session_id}/upload-screenshot")
async def legacy_upload_screenshot(session_id: str, screenshot_id: str = None, file: UploadFile = File(...)):
    print(f"LEGACY ENDPOINT: upload screenshot {session_id}")
    response = await upload_screenshot(session_id, screenshot_id or str(uuid.uuid4()), file)

    return {
        "status": "uploaded",
        "filename": response.filename,
        "screenshot_id": response.screenshotId,
        "url": response.url
    }


@app.get("/api/session/{session_id}/screenshot/{filename}")
async def legacy_get_screenshot(session_id: str, filename: str):
    return await get_screenshot(session_id, filename)


@app.get("/api/session/{session_id}/shadows")
async def legacy_load_shadows(session_id: str):
    print(f"LEGACY ENDPOINT: load shadows {session_id}")

    session_data = load_session(session_id)
    if not session_data:
        return {"status": "not_found", "data": None}

    if not session_data.shadows:
        return {"status": "not_found", "data": None}

    legacy_data = {
        "version": "2.0",
        "screenshots": [
            {
                "screenshotId": s.screenshotId,
                "objects": [
                    {
                        "id": obj.id,
                        "name": obj.name,
                        "pairs": [
                            {
                                "objectPoint": p.objectPoint.model_dump(),
                                "shadowPoint": p.shadowPoint.model_dump()
                            }
                            for p in obj.pairs
                        ]
                    }
                    for obj in s.objects
                ]
            }
            for s in session_data.shadows
        ]
    }

    return {"status": "found", "data": legacy_data}


@app.post("/api/session/{session_id}/shadows")
async def legacy_save_shadows(session_id: str, data: dict):
    print(f"LEGACY ENDPOINT: save shadows {session_id}")

    from models import ScreenshotShadows, ShadowObject, ShadowPair, NormalizedPoint2D, ShadowPoint

    session_data = load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session_data.shadows = [
        ScreenshotShadows(
            screenshotId=s.get('screenshotId', s.get('id', '')),
            objects=[
                ShadowObject(
                    id=obj.get('id', ''),
                    name=obj.get('name', ''),
                    pairs=[
                        ShadowPair(
                            objectPoint=NormalizedPoint2D(
                                normalizedX=p['objectPoint']['normalizedX'],
                                normalizedY=p['objectPoint']['normalizedY']
                            ),
                            shadowPoint=ShadowPoint(
                                normalizedX=p['shadowPoint']['normalizedX'],
                                normalizedY=p['shadowPoint']['normalizedY'],
                                wall=p['shadowPoint']['wall'],
                                world3D=Point3D(**p['shadowPoint']['world3D']) if p['shadowPoint'].get('world3D') else None
                            )
                        )
                        for p in obj.get('pairs', [])
                    ]
                )
                for obj in s.get('objects', [])
            ]
        )
        for s in data.get('screenshots', [])
    ]

    save_session(session_data)

    return {"status": "saved"}


@app.get("/api/sessions/{session_id}/shadows")
async def load_shadows_new(session_id: str):
    return await legacy_load_shadows(session_id)
