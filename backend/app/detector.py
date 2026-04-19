import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition
import dlib
from scipy.spatial import distance as dist
import os

class MaskAndFaceDetector:
    def __init__(self, mask_model_path=None):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        
        if mask_model_path is None:
            self.mask_model_path = os.path.join(self.data_dir, "mask_detector.keras")
        else:
            self.mask_model_path = mask_model_path
            
        self.face_net = None
        self.mask_net = None
        self.predictor = None # Dlib shape predictor
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Blink detection eye indices
        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42
        
        self.load_models()

    def load_models(self):
        # 1. Face Detector
        prototxtPath = os.path.join(self.data_dir, "face_detector", "deploy.prototxt")
        weightsPath = os.path.join(self.data_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(prototxtPath) and os.path.exists(weightsPath):
            self.face_net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

        # 2. Mask Model
        if os.path.exists(self.mask_model_path):
            self.mask_net = tf.keras.models.load_model(self.mask_model_path)

        # 3. Liveness Predictor (Dlib)
        landmark_path = os.path.join(self.data_dir, "shape_predictor_68_face_landmarks.dat")
        if os.path.exists(landmark_path):
            self.predictor = dlib.shape_predictor(landmark_path)
            print("[INFO] Liveness detection (blink) enabled.")

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def update_known_faces(self, db_users):
        self.known_face_encodings = []
        self.known_face_names = []
        for user in db_users:
            encoding = np.frombuffer(user.encoding, dtype=np.float64)
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(user.name)

    def detect_and_process(self, frame, eye_cache=None):
        if self.face_net is None: 
            return []
        
        if frame is None or frame.size == 0:
            return []

        (h, w) = frame.shape[:2]
        # SSD Face Detector works best with 300x300
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        try:
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
        except cv2.error as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []

        results = []
        rgb_frame = None
        gray = None

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5: # Slightly lower threshold for better recall, filtered later if needed
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Ensure box is valid
                if endX <= startX or endY <= startY:
                    continue

                # --- LIVENESS DETECTION (BLINK) ---
                is_live = False
                ear = 0.0
                if self.predictor:
                    if gray is None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    try:
                        shape = self.predictor(gray, rect)
                        shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                        
                        leftEye = shape_np[self.lStart:self.lEnd]
                        rightEye = shape_np[self.rStart:self.rEnd]
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        
                        is_live = ear > 0.15 
                    except Exception as e:
                        print(f"[WARNING] Liveness predictor failed: {e}")

                # --- MASK DETECTION ---
                mask_label = "Checking..."
                mask_conf = 0.0
                if self.mask_net:
                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.size > 0:
                        try:
                            face_roi = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB), (224, 224))
                            face_roi = img_to_array(face_roi)
                            face_roi = preprocess_input(face_roi)
                            preds = self.mask_net.predict(np.expand_dims(face_roi, axis=0), verbose=0)[0]
                            mask_label = "Mask" if preds[0] > preds[1] else "No Mask"
                            mask_conf = float(max(preds))
                        except Exception as e:
                            print(f"[ERROR] Mask detection failed: {e}")
                else:
                    mask_label = "System Ready" # If model not loaded but face found

                # --- IDENTITY ---
                name = "Unknown"
                if self.known_face_encodings:
                    if rgb_frame is None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    face_location = (startY, endX, endY, startX)
                    try:
                        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
                        if encodings:
                            matches = face_recognition.compare_faces(self.known_face_encodings, encodings[0], tolerance=0.45)
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = self.known_face_names[first_match_index]
                    except Exception as e:
                        print(f"[WARNING] Recognition failed: {e}")

                results.append({
                    "box": [int(startX), int(startY), int(endX), int(endY)],
                    "mask_status": mask_label,
                    "mask_confidence": mask_conf,
                    "name": name,
                    "is_live": is_live,
                    "ear": float(ear)
                })

        return results
