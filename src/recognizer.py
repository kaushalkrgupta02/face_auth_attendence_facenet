# src/recognizer.py
import torch
import numpy as np
import cv2
import os
from facenet_pytorch import InceptionResnetV1
from .config import DB_PATH, RECOGNITION_THRESHOLD

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize FaceNet
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.known_embeddings = {}
        self.load_db()

    def load_db(self):
        if os.path.exists(DB_PATH):
            try:
                self.known_embeddings = torch.load(DB_PATH)
                print(f"Loaded {len(self.known_embeddings)} users.")
            except:
                print("Database corrupted or empty. Starting fresh.")
                self.known_embeddings = {}
        else:
            self.known_embeddings = {}

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
        
        # --- THE FIX IS HERE ---
        # Explicitly cast to Python int() because OpenCV rejects NumPy int32/int64
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        center = (center_x, center_y)
        
        # Rotate whole frame
        try:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            # Fallback if rotation fails (e.g. extreme coords)
            print(f"Rotation failed: {e}")
            rotated_frame = frame

        # 2. Crop
        # Ensure we don't crop outside image bounds
        h, w = rotated_frame.shape[:2]
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(w, box[2])
        y2 = min(h, box[3])
        
        face_img = rotated_frame[y1:y2, x1:x2]
        
        # 3. Preprocess
        if face_img.size == 0: return None # Handle empty crops
        
        try:
            face_img = cv2.resize(face_img, (160, 160))
        except:
            return None 

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
        
        if len(self.known_embeddings) == 0:
            return "Unknown", 100

        for name, db_emb in self.known_embeddings.items():
            # Support both single embedding and averaged lists
            if isinstance(db_emb, list):
                # If you haven't averaged them yet in DB, take mean now
                db_emb = torch.stack(db_emb).mean(dim=0)
                
            dist = (embedding - db_emb).norm().item()
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        if min_dist > RECOGNITION_THRESHOLD:
            return "Unknown", min_dist
        
        return identity, min_dist

def register_face(self, name, samples):
        """
        Saves the MEAN (Average) of the collected samples.
        PREVENTS overwriting if user already exists.
        """
        if not samples: return False

        # User Existence ---
        if name in self.known_embeddings:
            print(f"\n[ERROR] Registration Failed: User '{name}' already exists in the database!")
            print("[HINT] To overwrite, delete 'data/face_db.pt' or use a different name.\n")
            return False
        
        # Stack list into a tensor: shape (5, 512)
        try:
            stacked = torch.cat(samples, dim=0)
            
            # Calculate Mean
            mean_embedding = torch.mean(stacked, dim=0, keepdim=True)
            
            # Save to dictionary
            self.known_embeddings[name] = mean_embedding
            
            # Save to disk
            self.save_db()
            return True
            
        except Exception as e:
            print(f"Registration Error: {e}")
            return False