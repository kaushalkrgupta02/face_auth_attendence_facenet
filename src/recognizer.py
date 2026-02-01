import torch
import numpy as np
import cv2
import os
from facenet_pytorch import InceptionResnetV1
from .config import DB_PATH, RECOGNITION_THRESHOLD

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.known_embeddings = {}
        self.load_db()

    def load_db(self):
        if os.path.exists(DB_PATH):
            self.known_embeddings = torch.load(DB_PATH)
            print(f"Loaded {len(self.known_embeddings)} users.")

    def save_db(self):
        torch.save(self.known_embeddings, DB_PATH)
        print("Database saved.")

    def get_embedding(self, frame, face_data):
        """
        Aligns face, crops, standardizes, and returns embedding.
        """
        box = face_data['facial_area']
        landmarks = face_data['landmarks']

        # 1. Alignment (Rotation based on eyes)
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Rotate whole frame
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        # 2. Crop
        face_img = rotated_frame[box[1]:box[3], box[0]:box[2]]
        
        # 3. Preprocess
        try:
            face_img = cv2.resize(face_img, (160, 160))
        except:
            return None # ROI might be out of bounds after rotation

        face_img = np.float32(face_img)
        face_img = (face_img - 127.5) / 128.0
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 4. Infer
        with torch.no_grad():
            emb = self.model(face_tensor).cpu()
        return emb

    def identify(self, embedding):
        min_dist = 100
        identity = "Unknown"
        
        for name, db_emb in self.known_embeddings.items():
            dist = (embedding - db_emb).norm().item()
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        if min_dist > RECOGNITION_THRESHOLD:
            return "Unknown", min_dist
        
        return identity, min_dist

    def register_face(self, name, embedding):
        self.known_embeddings[name] = embedding
        self.save_db()