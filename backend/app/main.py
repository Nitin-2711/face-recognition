from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = FastAPI(title="Mask Detection Minimal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
face_model_path = os.path.join(data_dir, "face_detector")
mask_model_path = os.path.join(data_dir, "mask_detector.keras")

face_net = None
mask_net = None

prototxtPath = os.path.join(face_model_path, "deploy.prototxt")
weightsPath = os.path.join(face_model_path, "res10_300x300_ssd_iter_140000.caffemodel")

try:
    if os.path.exists(prototxtPath) and os.path.exists(weightsPath):
        face_net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
        print("[INFO] Face detector loaded.")
    if os.path.exists(mask_model_path):
        mask_net = tf.keras.models.load_model(mask_model_path)
        print("[INFO] Mask detector loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"mask": False}

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        if face_net:
            face_net.setInput(blob)
            detections = face_net.forward()
        else:
            print("[WARN] Face net not loaded.")
            return {"mask": False}

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                if endX <= startX or endY <= startY:
                    continue

                if mask_net:
                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.size > 0:
                        face_roi = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB), (224, 224))
                        face_roi = img_to_array(face_roi)
                        face_roi = preprocess_input(face_roi)
                        preds = mask_net.predict(np.expand_dims(face_roi, axis=0), verbose=0)[0]
                        
                        is_mask = bool(preds[0] > preds[1])
                        return {"mask": is_mask}
        return {"mask": False}
    except Exception as e:
        print(f"[ERROR] Detection Error: {e}")
        return {"mask": False}
