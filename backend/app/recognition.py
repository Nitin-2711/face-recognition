import face_recognition
import numpy as np
import json
import cv2
import os
from .models import User
from sqlalchemy.orm import Session

class FaceHandler:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []

    def load_known_faces(self, db: Session):
        users = db.query(User).all()
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        for user in users:
            if user.face_encoding:
                encoding = np.array(json.loads(user.face_encoding))
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(user.name)
                self.known_face_ids.append(user.id)

    def recognize_face(self, frame_data):
        # frame_data is a numpy array from cv2
        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            user_id = None

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    user_id = self.known_face_ids[best_match_index]

            results.append({
                "name": name,
                "user_id": user_id,
                "location": (top, right, bottom, left)
            })
        
        return results

    def get_encoding(self, image_path):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            return encodings[0].tolist()
        return None

face_handler = FaceHandler()
