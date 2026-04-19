from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import base64
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import face_recognition
import os
import json
import logging

from .database import engine, get_db, Base
from .models import User, Attendance
from .detector import MaskAndFaceDetector

# Robust path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIOLATIONS_DIR = os.path.join(DATA_DIR, "violations")

os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Initialize Database
Base.metadata.create_all(bind=engine)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecuTrack")

app = FastAPI(title="Face Attendance & Mask System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files correctly
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# Initialize Detector
detector = MaskAndFaceDetector()

last_log_time = {} # Name -> Datetime

def refresh_faces():
    db = next(get_db())
    users = db.query(User).all()
    detector.update_known_faces(users)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting SecuTrack Server...")
    refresh_faces()

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(), "users_loaded": len(detector.known_face_names)}

@app.post("/users/register")
async def register_user(
    name: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        encodings = face_recognition.face_encodings(rgb_img)
        if not encodings:
            return {"status": "error", "message": "No face detected in profile image"}
        
        encoding = encodings[0].tobytes()
        
        existing = db.query(User).filter(User.name == name).first()
        if existing:
            existing.encoding = encoding
            logger.info(f"Updated encoding for user {name}")
        else:
            new_user = User(name=name, encoding=encoding)
            db.add(new_user)
            logger.info(f"Registered new user {name}")
        
        db.commit()
        refresh_faces()
        return {"status": "success", "message": f"User {name} registered"}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/stream")
async def process_stream(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    logger.info("Websocket Connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            if "," not in data: continue
            
            _, encoded = data.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue

            detections = detector.detect_and_process(frame)

            for det in detections:
                name = det["name"]
                mask = det["mask_status"]
                
                if name != "Unknown":
                    now = datetime.now()
                    should_log = False
                    
                    if name not in last_log_time or (now - last_log_time[name]) > timedelta(minutes=5): # Reduced for testing
                        should_log = True
                    
                    if mask == "No Mask":
                        should_log = True
                    
                    if should_log:
                        last_log_time[name] = now
                        save_attendance(db, name, mask, det["mask_confidence"], frame, det["box"])

            await websocket.send_json({
                "results": detections,
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        logger.info("Websocket Disconnected")
    except Exception as e:
        logger.error(f"Stream error: {e}")

def save_attendance(db: Session, name, mask, conf, frame, box):
    user = db.query(User).filter(User.name == name).first()
    if not user: return

    (startX, startY, endX, endY) = box
    roi = frame[startY:endY, startX:endX]
    filename = f"att_{name}_{datetime.now().strftime('%H%M%S')}.jpg"
    path = os.path.join(VIOLATIONS_DIR, filename)
    
    if roi.size > 0:
        cv2.imwrite(path, roi)
        logger.info(f"Attendance logged for {name} ({mask})")

    log = Attendance(
        user_id=user.id,
        mask_status=mask,
        confidence=conf,
        screenshot_path=f"/static/violations/{filename}"
    )
    db.add(log)
    db.commit()

@app.get("/attendance")
async def get_attendance(db: Session = Depends(get_db)):
    logs = db.query(Attendance).order_by(Attendance.timestamp.desc()).limit(100).all()
    return [{
        "id": l.id,
        "name": l.user.name if l.user else "Unknown",
        "timestamp": l.timestamp,
        "mask_status": l.mask_status,
        "confidence": l.confidence,
        "image": l.screenshot_path
    } for l in logs]

@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "name": u.name, "created_at": u.created_at} for u in users]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
