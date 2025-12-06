"""
Shadow Geolocation Backend

VERSION 3.0:
- Zentrale session.json (alle Daten in einer Datei)
- Vereinfachte API (4 Endpoints statt 8+)
- Kein Legacy v1/v2 Support
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import json
from pathlib import Path
import uuid
from datetime import datetime

from models import (
    SessionData, CreateSessionRequest, CreateSessionResponse,
    UploadScreenshotResponse, MetaData, CalibrationData,
    RoomDimensions, CameraParams, Point3D, ScreenshotCalibration,
    EulerRotation, DisplayParams, BundleAdjustmentRequest
)

from solver.bundle_adjustment import bundle_adjustment_async

# FastAPI App
app = FastAPI(
    title="Shadow Geolocation Backend",
    version="3.0",
    docs_url="/docs",  # Swagger UI aktiviert für Entwicklung
    redoc_url="/redoc"
)

# CORS für Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File-Storage
DATA_DIR = Path("data/uploads")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_session_dir(session_id: str) -> Path:
    """Gibt den Session-Ordner zurück"""
    return DATA_DIR / session_id


def get_session_file(session_id: str) -> Path:
    """Gibt den Pfad zur session.json zurück"""
    return get_session_dir(session_id) / "session.json"


def get_screenshots_dir(session_id: str) -> Path:
    """Gibt den Screenshots-Ordner zurück"""
    return get_session_dir(session_id) / "screenshots"


def load_session(session_id: str) -> Optional[SessionData]:
    """Lädt eine Session aus der session.json"""
    session_file = get_session_file(session_id)
    
    if not session_file.exists():
        return None
    
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return SessionData(**data)


def save_session(session_data: SessionData) -> None:
    """Speichert eine Session in die session.json"""
    session_file = get_session_file(session_data.sessionId)
    
    # LastModified aktualisieren
    session_data.meta.lastModified = datetime.utcnow().isoformat() + "Z"
    
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"💾 Session gespeichert: {session_data.sessionId}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health Check"""
    return {"status": "Shadow Geolocation Backend v3.0 running"}


@app.get("/health")
async def health():
    """Health Check"""
    return {"status": "ok", "version": "3.0"}


# ----------------------------------------------------------------------------
# SESSION CRUD
# ----------------------------------------------------------------------------

@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Erstellt eine neue Session.
    
    Wird von Stage 1 aufgerufen nach Eingabe von Projektname und Screenshots.
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    
    # Session-Ordner anlegen
    session_dir = get_session_dir(session_id)
    session_dir.mkdir(exist_ok=True)
    get_screenshots_dir(session_id).mkdir(exist_ok=True)
    
    # Session-Daten erstellen
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
    
    # Speichern
    save_session(session_data)
    
    print(f"✅ Neue Session erstellt: {session_id}")
    print(f"   Projekt: {request.projectName}")
    print(f"   Screenshots: {len(request.screenshots)}")
    
    return CreateSessionResponse(
        sessionId=session_id,
        projectName=request.projectName
    )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Lädt eine komplette Session.
    
    Wird von allen Stages beim Initialisieren aufgerufen.
    """
    session_data = load_session(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    
    print(f"📂 Session geladen: {session_id}")
    print(f"   Projekt: {session_data.meta.projectName}")
    print(f"   Screenshots: {len(session_data.screenshots)}")
    print(f"   Calibration: {'✓' if session_data.calibration else '✗'}")
    print(f"   Shadows: {'✓' if session_data.shadows else '✗'}")
    
    return session_data.model_dump()


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, session_data: SessionData):
    """
    Speichert eine komplette Session.
    
    Wird beim Speichern (manuell oder bei Navigation) aufgerufen.
    """
    # Prüfe ob Session existiert
    if not get_session_dir(session_id).exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    
    # Session-ID muss übereinstimmen
    if session_data.sessionId != session_id:
        raise HTTPException(status_code=400, detail="Session-ID stimmt nicht überein")
    
    # Speichern
    save_session(session_data)
    
    print(f"💾 Session aktualisiert: {session_id}")
    
    return {"status": "saved", "sessionId": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Löscht eine Session (optional, für später).
    """
    session_dir = get_session_dir(session_id)
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    
    # Rekursiv löschen
    import shutil
    shutil.rmtree(session_dir)
    
    print(f"🗑️ Session gelöscht: {session_id}")
    
    return {"status": "deleted", "sessionId": session_id}


# ----------------------------------------------------------------------------
# SCREENSHOTS
# ----------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/screenshots", response_model=UploadScreenshotResponse)
async def upload_screenshot(
    session_id: str,
    screenshot_id: str,
    file: UploadFile = File(...)
):
    """
    Lädt einen Screenshot hoch.
    
    Wird von Stage 1 für jeden Screenshot aufgerufen.
    """
    session_dir = get_session_dir(session_id)
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    
    screenshots_dir = get_screenshots_dir(session_id)
    screenshots_dir.mkdir(exist_ok=True)
    
    # Dateiname: screenshot_id.png
    filename = f"{screenshot_id}.png"
    file_path = screenshots_dir / filename
    
    # Speichern
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    print(f"📸 Screenshot hochgeladen: {filename} ({len(content)} bytes)")
    
    return UploadScreenshotResponse(
        filename=filename,
        screenshotId=screenshot_id,
        url=f"/api/sessions/{session_id}/screenshots/{filename}"
    )


@app.get("/api/sessions/{session_id}/screenshots/{filename}")
async def get_screenshot(session_id: str, filename: str):
    """
    Gibt einen Screenshot zurück.
    """
    file_path = get_screenshots_dir(session_id) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot nicht gefunden")
    
    return FileResponse(file_path, media_type="image/png")


# ============================================================================
# BUNDLE ADJUSTMENT (WebSocket)
# ============================================================================

@app.websocket("/ws/bundle-adjustment")
async def websocket_bundle_adjustment(websocket: WebSocket):
    """
    WebSocket für Bundle Adjustment mit Live-Progress.
    
    VERSION 3.0:
    - Verwendet neues SessionData/CalibrationData Format
    - Kein Legacy-Support mehr
    """
    await websocket.accept()
    print("✅ WebSocket connected (Bundle Adjustment)")
    
    try:
        # Request empfangen
        raw_data = await websocket.receive_json()
        
        print(f"📦 Bundle Adjustment Request erhalten")
        print(f"   Session: {raw_data.get('sessionId', 'N/A')}")
        
        # Validiere Request
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
        
        print(f"   Room: {calibration_data['room']}")
        print(f"   Camera: {calibration_data['camera']}")
        print(f"   Screenshots: {len(calibration_data.get('screenshots', []))}")
        print(f"   Weights: {weights}")
        
        # Bundle Adjustment ausführen
        async for update in bundle_adjustment_async(calibration_data, weights):
            await websocket.send_json(update)
            
            if update['type'] in ['error', 'result']:
                print(f"✅ Bundle Adjustment beendet: {update['type']}")
                break
        
    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
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
            print("🔌 WebSocket closed")
        except:
            pass


# ============================================================================
# LEGACY ENDPOINTS (für Migration - später entfernen)
# ============================================================================

# Diese Endpoints werden temporär beibehalten, falls alte Clients sie noch nutzen.
# TODO: Nach vollständiger Migration entfernen

@app.post("/api/session/create")
async def legacy_create_session(data: dict):
    """LEGACY: Alte Session-Erstellung"""
    print("⚠️ LEGACY ENDPOINT AUFGERUFEN: /api/session/create")
    
    from models import ScreenshotData
    
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
    """LEGACY: Kalibrierung laden"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: /api/session/{session_id}/calibration")
    
    session_data = load_session(session_id)
    if not session_data:
        return {"status": "not_found", "data": None}
    
    if not session_data.calibration:
        return {"status": "not_found", "data": None}
    
    # Konvertiere zu altem Format
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
    """LEGACY: Kalibrierung speichern"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: POST /api/session/{session_id}/calibration")
    
    session_data = load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    
    # Konvertiere von altem Format
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
    """LEGACY: Organization laden"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: /api/session/{session_id}/organize")
    
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
    """LEGACY: Organization speichern"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: POST /api/session/{session_id}/organize")
    return {"status": "saved"}


@app.post("/api/session/{session_id}/upload-screenshot")
async def legacy_upload_screenshot(
    session_id: str,
    screenshot_id: str = None,
    file: UploadFile = File(...)
):
    """LEGACY: Screenshot hochladen"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: /api/session/{session_id}/upload-screenshot")
    
    response = await upload_screenshot(session_id, screenshot_id or str(uuid.uuid4()), file)
    
    return {
        "status": "uploaded",
        "filename": response.filename,
        "screenshot_id": response.screenshotId,
        "url": response.url
    }


@app.get("/api/session/{session_id}/screenshot/{filename}")
async def legacy_get_screenshot(session_id: str, filename: str):
    """LEGACY: Screenshot abrufen"""
    return await get_screenshot(session_id, filename)


@app.get("/api/session/{session_id}/shadows")
async def legacy_load_shadows(session_id: str):
    """LEGACY: Shadows laden"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: /api/session/{session_id}/shadows")
    
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
    """LEGACY: Shadows speichern"""
    print(f"⚠️ LEGACY ENDPOINT AUFGERUFEN: POST /api/session/{session_id}/shadows")
    
    from models import ScreenshotShadows, ShadowObject, ShadowPair, NormalizedPoint2D, ShadowPoint
    
    session_data = load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    
    # Konvertiere von altem Format
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


# Alias für sessions (ohne s am Ende)
@app.get("/api/sessions/{session_id}/shadows")
async def load_shadows_new(session_id: str):
    """Shadows laden (neuer Endpoint)"""
    return await legacy_load_shadows(session_id)
