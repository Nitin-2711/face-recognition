from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import base64
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import face_recognition
import os
import logging
from typing import Optional

from .database import engine, get_db, Base
from .models import User, Attendance
from .detector import MaskAndFaceDetector

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIOLATIONS_DIR = os.path.join(DATA_DIR, "violations")

os.makedirs(VIOLATIONS_DIR, exist_ok=True)
Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecuTrack")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing SecuTrack AI Core...")
    refresh_faces()
    yield
    # Shutdown
    logger.info("Shutting down SecuTrack AI Core...")

app = FastAPI(title="SecuTrack AI Production", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

detector = MaskAndFaceDetector()
last_log_time = {} 
session_liveness = {} 

def refresh_faces():
    try:
        db = next(get_db())
        users = db.query(User).all()
        detector.update_known_faces(users)
        logger.info(f"Loaded {len(users)} known identities.")
    except Exception as e:
        logger.error(f"Failed to refresh faces: {e}")

@app.get("/health")
async def health():
    return {
        "status": "online", 
        "liveness_ready": detector.predictor is not None,
        "mask_engine": "loaded" if detector.mask_net else "unavailable"
    }

@app.websocket("/ws/stream")
async def process_stream(websocket: WebSocket, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    await websocket.accept()
    logger.info("New biometric stream established.")
    try:
        while True:
            data = await websocket.receive_text()
            
            # Robust extraction of base64
            if "base64," in data:
                _, encoded = data.split("base64,", 1)
            elif "," in data:
                _, encoded = data.split(",", 1)
            else:
                encoded = data

            try:
                # Optimized decoding
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    continue

                detections = detector.detect_and_process(frame)

                for det in detections:
                    name = det["name"]
                    if name == "Unknown": continue
                    
                    # Require liveness verification before logging
                    if det["ear"] < 0.20:
                        session_liveness[name] = True
                        
                    is_live_verified = session_liveness.get(name, False)

                    if is_live_verified:
                        now = datetime.now()
                        mask = det["mask_status"]
                        
                        # Throttle logging per person
                        if name not in last_log_time or (now - last_log_time[name]) > timedelta(minutes=15):
                            last_log_time[name] = now
                            background_tasks.add_task(save_attendance, name, mask, det["mask_confidence"], frame, det["box"])

                await websocket.send_json({
                    "results": detections,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_json({"error": "Failed to process frame"})
                
    except WebSocketDisconnect:
        logger.info("Biometric stream disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

def save_attendance(name, mask, conf, frame, box):
    # This runs in background to prevent video lag
    db = next(get_db())
    user = db.query(User).filter(User.name == name).first()
    if not user: return

    (startX, startY, endX, endY) = box
    roi = frame[startY:endY, startX:endX]
    filename = f"att_{name}_{datetime.now().strftime('%H%M%S')}.jpg"
    path = os.path.join(VIOLATIONS_DIR, filename)
    
    if roi.size > 0:
        cv2.imwrite(path, roi)

    log = Attendance(
        user_id=user.id,
        mask_status=mask,
        confidence=conf,
        screenshot_path=f"/static/violations/{filename}"
    )
    db.add(log)
    db.commit()
    db.close()

@app.get("/attendance")
async def get_attendance(
    page: int = Query(1, ge=1), 
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    # Pagination: Production standard
    offset = (page - 1) * limit
    total = db.query(Attendance).count()
    logs = db.query(Attendance).order_by(Attendance.timestamp.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "data": [{
            "id": l.id,
            "name": l.user.name if l.user else "Unknown",
            "timestamp": l.timestamp,
            "mask_status": l.mask_status,
            "image": l.screenshot_path
        } for l in logs]
    }

@app.post("/detect")
async def detect_mask(file: UploadFile = File(...)):
    logger.info("Received frame for real-time analysis.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"mask": False, "confidence": 0.0, "error": "Invalid image"}

    results = detector.detect_and_process(frame)
    
    if len(results) > 0:
        # Return the first detection for simplicity in this mode
        res = results[0]
        return {
            "mask": res["mask_status"] == "Mask",
            "confidence": res["mask_confidence"]
        }
    
    return {"mask": False, "confidence": 0.0, "message": "No face detected"}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return []
    return detector.detect_and_process(frame)

@app.post("/log-attendance")
async def log_manual_attendance(
    background_tasks: BackgroundTasks,
    user_id: Optional[int] = Form(None),
    status: str = Form("Recognized"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    # Retrieve user or handle unknown
    name = "Unknown"
    if user_id:
        user = db.query(User).get(user_id)
        if user: name = user.name
    
    # Placeholder box for manual log
    box = [0, 0, 100, 100] 
    
    # Process image if provided, else dummy
    frame = None
    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

    save_attendance(name, "Scanning", 0.0, frame, box)
    return {"status": "success", "message": f"Attendance logged for {name}"}

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_users = db.query(User).count()
    
    # Today's attendance
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    present_today = db.query(Attendance).filter(Attendance.timestamp >= today_start).distinct(Attendance.user_id).count()
    
    # Unknown detections (those without a user_id or where we log unknown)
    unknown_detections = db.query(Attendance).filter(Attendance.user_id == None).count()
    
    attendance_rate = (present_today / total_users * 100) if total_users > 0 else 0
    
    return {
        "total_users": total_users,
        "present_today": present_today,
        "unknown_detections": unknown_detections,
        "attendance_rate": attendance_rate
    }

@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "name": u.name} for u in users]

@app.post("/users/register")
async def register_user(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    if not encodings: return {"status": "error", "message": "No face detected"}
    
    existing = db.query(User).filter(User.name == name).first()
    if existing: existing.encoding = encodings[0].tobytes()
    else: db.add(User(name=name, encoding=encodings[0].tobytes()))
    db.commit()
    refresh_faces()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
