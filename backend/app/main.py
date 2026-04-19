from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI(title="Minimal Mask Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal Detector Class
class SimpleMaskDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        
        # Face Detector
        prototxt = os.path.join(data_dir, "face_detector", "deploy.prototxt")
        weights = os.path.join(data_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        
        # Mask Detector
        mask_model_path = os.path.join(data_dir, "mask_detector.keras")
        self.mask_net = tf.keras.models.load_model(mask_model_path)

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                if face.size > 0:
                    face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    preds = self.mask_net.predict(np.expand_dims(face, axis=0), verbose=0)[0]
                    return preds[0] > preds[1] # True if Mask
        return False

detector = SimpleMaskDetector()

@app.get("/health")
async def health():
    return {"status": "online"}

@app.post("/detect")
async def detect_mask(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"mask": False}

    is_mask = detector.detect(frame)
    return {"mask": is_mask}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
