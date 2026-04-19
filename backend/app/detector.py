import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition
import os

class MaskAndFaceDetector:
    def __init__(self, mask_model_path=None):
        # Resolve absolute paths relative to backend root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        
        if mask_model_path is None:
            self.mask_model_path = os.path.join(self.data_dir, "mask_detector.model")
        else:
            self.mask_model_path = mask_model_path
            
        self.face_net = None
        self.mask_net = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_models()

    def load_models(self):
        # 1. Load Face Detector (SSD)
        prototxtPath = os.path.join(self.data_dir, "face_detector", "deploy.prototxt")
        weightsPath = os.path.join(self.data_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
        
        if os.path.exists(prototxtPath) and os.path.exists(weightsPath):
            print(f"[INFO] Loading face detector from {prototxtPath}")
            self.face_net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
        else:
            print(f"[CRITICAL] Face detector files missing at {prototxtPath}. Please run setup.py.")

        # 2. Load Mask Classification Model
        if os.path.exists(self.mask_model_path):
            try:
                self.mask_net = tf.keras.models.load_model(self.mask_model_path)
                print("[INFO] Mask model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Mask model load failed: {e}")
        else:
            print(f"[WARNING] No mask model found at {self.mask_model_path}. Running without mask detection.")

    def update_known_faces(self, db_users):
        self.known_face_encodings = []
        self.known_face_names = []
        for user in db_users:
            encoding = np.frombuffer(user.encoding, dtype=np.float64)
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(user.name)
        print(f"[INFO] Initialized {len(self.known_face_names)} authorized users.")

    def detect_and_process(self, frame):
        if self.face_net is None: return []
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (160, 160), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Mask Detection
                mask_label = "Scanning"
                mask_conf = 0.0
                if self.mask_net:
                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.size > 0:
                        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        face_roi = cv2.resize(face_roi, (224, 224))
                        face_roi = img_to_array(face_roi)
                        face_roi = preprocess_input(face_roi)
                        
                        preds = self.mask_net.predict(np.expand_dims(face_roi, axis=0), verbose=0)[0]
                        (mask, withoutMask) = preds
                        mask_label = "Mask" if mask > withoutMask else "No Mask"
                        mask_conf = float(max(mask, withoutMask))

                # Identity Detection
                face_location = (startY, endX, endY, startX)
                name = "Unknown"
                if self.known_face_encodings:
                    encodings = face_recognition.face_encodings(rgb_frame, [face_location])
                    if encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, encodings[0], tolerance=0.5)
                        if True in matches:
                            name = self.known_face_names[matches.index(True)]

                results.append({
                    "box": [int(startX), int(startY), int(endX), int(endY)],
                    "mask_status": mask_label,
                    "mask_confidence": mask_conf,
                    "name": name
                })

        return results
