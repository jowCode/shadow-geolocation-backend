from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import json
from pathlib import Path
import uuid

# Importiere die neuen Models und Bundle Adjustment
from models import (
    SessionCreate, SessionResponse, SolveRequest,
    BundleAdjustmentRequest, CalibrationScreenshotData
)
from bundle_adjustment import (
    bundle_adjustment_async, CalibrationData, CalibrationScreenshot,
    convert_legacy_request
)

# FastAPI OHNE Swagger
app = FastAPI(
    title="Shadow Geolocation Backend",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
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
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============== Endpoints ==============

@app.get("/")
async def root():
    return {"status": "Shadow Geolocation Backend running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/session/create", response_model=SessionResponse)
async def create_session(data: SessionCreate):
    """Erstelle neue Session"""
    session_id = str(uuid.uuid4())
    
    # Session-Ordner anlegen
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Metadaten speichern
    metadata = {
        "session_id": session_id,
        "project_name": data.project_name,
        "camera_type": data.camera_type
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return SessionResponse(
        session_id=session_id,
        project_name=data.project_name
    )


@app.post("/api/session/{session_id}/calibration")
async def save_calibration(session_id: str, data: dict):
    """Kalibrierungs-Daten speichern"""
    session_dir = UPLOAD_DIR / session_id
    
    if not session_dir.exists():
        return {"error": "Session not found"}, 404
    
    # Speichere als JSON
    calibration_file = session_dir / "calibration.json"
    with open(calibration_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Calibration saved for session {session_id}")
    print(f"   Version: {data.get('version', '1.0')}")
    print(f"   Completed screenshots: {sum(1 for s in data.get('screenshots', []) if s.get('completed'))}")
    
    return {"status": "saved", "file": str(calibration_file)}


@app.get("/api/session/{session_id}/calibration")
async def load_calibration(session_id: str):
    """Kalibrierungs-Daten laden"""
    session_dir = UPLOAD_DIR / session_id
    calibration_file = session_dir / "calibration.json"
    
    if not calibration_file.exists():
        print(f"⚠️  No calibration found for session {session_id}")
        return {"status": "not_found", "data": None}
    
    with open(calibration_file, "r") as f:
        data = json.load(f)
    
    print(f"📂 Calibration loaded for session {session_id}")
    print(f"   Version: {data.get('version', '1.0')}")
    print(f"   Completed screenshots: {sum(1 for s in data.get('screenshots', []) if s.get('completed'))}")
    
    return {"status": "found", "data": data}


@app.post("/api/session/{session_id}/shadows")
async def save_shadows(session_id: str, data: dict):
    """Schatten-Daten speichern"""
    session_dir = UPLOAD_DIR / session_id
    
    with open(session_dir / "shadows.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Shadows saved for session {session_id}")
    print(f"   Version: {data.get('version', '1.0')}")
    
    return {"status": "saved"}


@app.get("/api/session/{session_id}/shadows")
async def load_shadows(session_id: str):
    """Schatten-Daten laden"""
    session_dir = UPLOAD_DIR / session_id
    shadows_file = session_dir / "shadows.json"
    
    if not shadows_file.exists():
        return {"status": "not_found", "data": None}
    
    with open(shadows_file, "r") as f:
        data = json.load(f)
    
    return {"status": "found", "data": data}


@app.post("/api/session/{session_id}/organize")
async def save_organization(session_id: str, data: dict):
    """Screenshot-Organisation speichern"""
    session_dir = UPLOAD_DIR / session_id
    
    with open(session_dir / "organization.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"📋 Organization saved for session {session_id}")
    
    return {"status": "saved"}


@app.get("/api/session/{session_id}/organize")
async def load_organization(session_id: str):
    """Screenshot-Organisation laden"""
    session_dir = UPLOAD_DIR / session_id
    organization_file = session_dir / "organization.json"
    
    if not organization_file.exists():
        print(f"⚠️  No organization found for session {session_id}")
        return {"status": "not_found", "data": None}
    
    with open(organization_file, "r") as f:
        data = json.load(f)
    
    print(f"📂 Organization loaded for session {session_id}")
    print(f"   Screenshots: {len(data.get('screenshots', []))}")
    
    return {"status": "found", "data": data}


@app.post("/api/session/{session_id}/upload-screenshot")
async def upload_screenshot(session_id: str, screenshot_id: str = None, file: UploadFile = File(...)):
    """Screenshot hochladen und speichern"""
    session_dir = UPLOAD_DIR / session_id
    screenshots_dir = session_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    
    # Dateiname: screenshot_id oder original filename
    if screenshot_id:
        filename = f"{screenshot_id}.png"
    else:
        filename = file.filename
    
    file_path = screenshots_dir / filename
    
    # Speichern
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    print(f"📸 Screenshot saved: {filename} ({len(content)} bytes)")
    
    return {
        "status": "uploaded",
        "filename": filename,
        "screenshot_id": screenshot_id or filename.split('.')[0],
        "url": f"/api/session/{session_id}/screenshot/{filename}"
    }


@app.get("/api/session/{session_id}/screenshot/{filename}")
async def get_screenshot(session_id: str, filename: str):
    """Screenshot abrufen"""
    file_path = UPLOAD_DIR / session_id / "screenshots" / filename
    
    if not file_path.exists():
        return {"error": "Screenshot not found"}, 404
    
    return FileResponse(file_path)


@app.websocket("/ws/bundle-adjustment")
async def websocket_bundle_adjustment(websocket: WebSocket):
    """
    WebSocket für Bundle Adjustment mit Live-Progress.
    
    VERSION 2.0:
    - Unterstützt sowohl v1.0 als auch v2.0 Request-Format
    - Verwendet convert_legacy_request() für Kompatibilität
    
    Client sendet BundleAdjustmentRequest als JSON.
    Server sendet Progress-Updates und final das Ergebnis.
    """
    await websocket.accept()
    print("✅ WebSocket accepted")
    
    try:
        # Empfange Request-Daten
        print("⏳ Waiting for request data...")
        raw_data = await websocket.receive_json()
        
        version = raw_data.get('version', '1.0')
        print(f"📦 Received data (version {version})")
        print(f"   Screenshots: {len(raw_data.get('screenshots', []))}")
        
        # Konvertiere zu CalibrationData (unterstützt v1.0 und v2.0)
        calibration_data = convert_legacy_request(raw_data)
        
        print(f"🔧 Converted to CalibrationData:")
        print(f"   Room: {calibration_data.room}")
        print(f"   Camera: {calibration_data.camera_position}")
        print(f"   FOV: {calibration_data.fov_y}°")
        print(f"   Screenshots: {len(calibration_data.screenshots)}")
        
        # Weights extrahieren
        weights = raw_data.get('weights', {
            'room_confidence': 0.5,
            'position_confidence': 0.5
        })
        print(f"   Weights: {weights}")
        
        # Starte Bundle Adjustment mit Progress-Updates
        update_count = 0
        async for update in bundle_adjustment_async(calibration_data, weights):
            update_count += 1
            print(f"📤 Sending update #{update_count}: {update['type']} - {update.get('message', '')}")
            await websocket.send_json(update)
            
            # Bei Fehler oder Ergebnis: Verbindung schließen
            if update['type'] in ['error', 'result']:
                print(f"✅ Bundle Adjustment finished: {update['type']}")
                break
        
        print(f"📊 Total updates sent: {update_count}")
        
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


@app.websocket("/ws/solve")
async def websocket_solve(websocket: WebSocket):
    """WebSocket für Standort-Berechnung mit Live-Progress"""
    await websocket.accept()
    
    try:
        # Empfange Request-Daten
        data = await websocket.receive_json()
        request = SolveRequest(**data)
        
        # Importiere Solver (wird später implementiert)
        # from solver.grid_search import solve_location_async
        
        # Placeholder
        await websocket.send_json({
            "type": "error",
            "message": "Solve-Funktion noch nicht implementiert"
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()