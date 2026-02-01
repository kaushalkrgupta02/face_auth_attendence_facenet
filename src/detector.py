import cv2
import numpy as np
from retinaface import RetinaFace
from .config import RETINA_CONFIDENCE, MIN_FACE_WIDTH, ROI_CENTER_PCT, GAZE_THRESHOLD_LOW, GAZE_THRESHOLD_HIGH

class FaceDetector:
    def detect(self, frame):
        """
        Wrapper for RetinaFace. 
        Returns a dictionary of faces or empty dict.
        """
        # RetinaFace returns a dict: {'face_1': {'score': ..., 'facial_area': ..., 'landmarks': ...}}
        return RetinaFace.detect_faces(frame)

    def verify_intent(self, face_data, frame_width, frame_height):
        """
        Filters out passersby using 3 checks:
        1. ROI (Magic Box)
        2. Size (Distance)
        3. Gaze (Head Pose)
        """
        box = face_data['facial_area'] # [x1, y1, x2, y2]
        landmarks = face_data['landmarks']
        score = face_data['score']

        if score < RETINA_CONFIDENCE:
            return False, "Low Conf"

        # 1. ROI Check (Magic Box)
        # Check if face center is inside the central box
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        roi_x1 = int(frame_width * (0.5 - ROI_CENTER_PCT/2))
        roi_x2 = int(frame_width * (0.5 + ROI_CENTER_PCT/2))
        roi_y1 = int(frame_height * 0.2)
        roi_y2 = int(frame_height * 0.8)

        if not (roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2):
            return False, "Step in Box"

        # 2. Size Check (Distance)
        face_width = x2 - x1
        if face_width < MIN_FACE_WIDTH:
            return False, "Come Closer"

        # 3. Gaze Check (Head Pose)
        # Check if nose is between eyes
        left_eye = landmarks['left_eye'][0]
        right_eye = landmarks['right_eye'][0]
        nose = landmarks['nose'][0]
        
        eye_dist = right_eye - left_eye
        if eye_dist == 0: return False, "Angle Err"
        
        nose_ratio = (nose - left_eye) / eye_dist
        if nose_ratio < GAZE_THRESHOLD_LOW or nose_ratio > GAZE_THRESHOLD_HIGH:
            return False, "Look Straight"

        return True, "Verified"

    def draw_roi(self, frame):
        """Draws the white box to guide the user."""
        h, w, _ = frame.shape
        roi_x1 = int(w * (0.5 - ROI_CENTER_PCT/2))
        roi_x2 = int(w * (0.5 + ROI_CENTER_PCT/2))
        roi_y1 = int(h * 0.2)
        roi_y2 = int(h * 0.8)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)