# src/liveness.py
import cv2
import numpy as np
from collections import deque
from core.config import BLINK_THRESHOLD, BLINK_FRAMES, HEAD_POSE_THRESHOLD, LIVENESS_CHALLENGE_TIMEOUT
import time
import random

class LivenessDetector:
    """
    Detects liveness through eye blink and head pose challenges.
    Prevents spoofing via photos/videos.
    """
    
    def __init__(self):
        self.blink_history = deque(maxlen=BLINK_FRAMES)
        self.challenge_active = False
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_passed = False
        self.blink_counter = 0  # Track consecutive blinks
        self.failed_attempts = 0
        
    def get_random_challenge(self):
        """Returns random liveness challenge"""
        return random.choice(['blink', 'look_left', 'look_right'])
    
    def start_challenge(self):
        """Initialize a new liveness challenge"""
        self.current_challenge = self.get_random_challenge()
        self.challenge_active = True
        self.challenge_start_time = time.time()
        self.challenge_passed = False
        self.blink_history.clear()
        self.blink_counter = 0
        return self.current_challenge
    
    def check_timeout(self):
        """Check if challenge exceeded timeout"""
        if not self.challenge_active:
            return False
        assert self.challenge_start_time is not None
        elapsed = time.time() - self.challenge_start_time
        if elapsed > LIVENESS_CHALLENGE_TIMEOUT:
            self.challenge_active = False
            return True
        return False
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Eye landmarks structure (6 points):
            1 (top-left)
        0               2 (top-right)
        
        5               3 (bottom-right)
            4 (bottom-left)
        
        Args:
            eye_landmarks: numpy array of shape (6, 2) with [x, y] coordinates
        
        Returns:
            ear: float, Eye Aspect Ratio
        """
        if len(eye_landmarks) != 12:
            return 1.0
        
        # Convert to numpy array if needed
        eye_landmarks = np.array(eye_landmarks).reshape(6, 2)
        
        # Vertical distances (eye height)
        # Distance from top-left to bottom-left
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        # Distance from top-right to bottom-right
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance (eye width)
        # Distance from left corner to right corner
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR = (vertical_1 + vertical_2) / (2 * horizontal)
        # When eyes open: higher EAR (~0.4-0.5)
        # When eyes closed: lower EAR (~0.1-0.2)
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal > 0 else 1.0
        return ear
    
    def detect_blink(self, landmarks):
        """
        Detect if user is blinking by tracking EAR changes.
        
        Blink detection logic:
        1. Calculate EAR for both eyes
        2. If EAR < BLINK_THRESHOLD: eyes are closed
        3. Track closure pattern: open -> close -> open
        
        Args:
            landmarks: dict with eye landmarks from RetinaFace
        
        Returns:
            bool: True if complete blink detected
        """
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return False
        
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Determine if eyes are closed
        eyes_closed = avg_ear < BLINK_THRESHOLD
        
        # Store in history (deque keeps last BLINK_FRAMES entries)
        self.blink_history.append(eyes_closed)
        
        # Detect blink pattern: closed -> open OR open -> closed
        if len(self.blink_history) == BLINK_FRAMES:
            # Check for transition from open to closed to open
            # Example pattern: [False, True, True, False] = blink detected
            has_closed = any(self.blink_history)      # At least one frame closed
            has_open = not all(self.blink_history)    # At least one frame open
            blink_detected = has_closed and has_open
            
            if blink_detected:
                self.blink_counter += 1
            
            return blink_detected
        
        return False
    
    def get_eye_metrics(self, landmarks):
        """
        Calculate and return eye metrics for visualization.
        
        Returns:
            dict with left_ear, right_ear, avg_ear, and eyes_closed status
        """
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return None
        
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'eyes_closed': avg_ear < BLINK_THRESHOLD
        }
    
    def draw_eye_landmarks(self, frame, landmarks):
        """
        Draw eye landmarks on frame for visualization.
        
        Shows:
        - Green circles for eye corner points
        - Yellow lines connecting the points
        - EAR value displayed
        """
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return
        
        if len(landmarks['left_eye']) != 12 or len(landmarks['right_eye']) != 12:
            return
        
        left_eye = np.array(landmarks['left_eye']).reshape(6, 2).astype(np.int32)
        right_eye = np.array(landmarks['right_eye']).reshape(6, 2).astype(np.int32)
        
        # Draw left eye
        for point in left_eye:
            cv2.circle(frame, tuple(point), 3, (0, 255, 0), -1)  # Green circles
        
        # Connect points to show eye shape
        for i in range(6):
            p1 = left_eye[i]
            p2 = left_eye[(i + 1) % 6]
            cv2.line(frame, tuple(p1), tuple(p2), (255, 255, 0), 1)  # Yellow lines
        
        # Draw right eye
        for point in right_eye:
            cv2.circle(frame, tuple(point), 3, (0, 255, 0), -1)  # Green circles
        
        # Connect points to show eye shape
        for i in range(6):
            p1 = right_eye[i]
            p2 = right_eye[(i + 1) % 6]
            cv2.line(frame, tuple(p1), tuple(p2), (255, 255, 0), 1)  # Yellow lines
        
        # Display EAR metrics
        metrics = self.get_eye_metrics(landmarks)
        if metrics:
            text = f"L:{metrics['left_ear']:.2f} R:{metrics['right_ear']:.2f} A:{metrics['avg_ear']:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show threshold line
            threshold_text = f"Threshold: {BLINK_THRESHOLD:.2f}"
            if metrics['eyes_closed']:
                cv2.putText(frame, f"{threshold_text} [CLOSED]", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"{threshold_text} [OPEN]", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def detect_head_pose(self, landmarks):
        """
        Detect head turn direction (left/right).
        Returns 'left', 'right', or 'center'
        """
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks or 'nose' not in landmarks:
            return 'center'
        
        if len(landmarks['left_eye']) != 12 or len(landmarks['right_eye']) != 12 or len(landmarks['nose']) != 2:
            return 'center'
        
        left_eye_points = np.array(landmarks['left_eye']).reshape(6, 2)
        right_eye_points = np.array(landmarks['right_eye']).reshape(6, 2)
        nose = np.array(landmarks['nose'])
        
        left_eye = left_eye_points[0]
        right_eye = right_eye_points[0]
        
        # Normalize nose position between eyes
        eye_distance = right_eye - left_eye
        if eye_distance == 0:
            return 'center'
        
        nose_ratio = (nose - left_eye) / eye_distance
        
        # Determine head direction
        if nose_ratio < (0.5 - HEAD_POSE_THRESHOLD):
            return 'left'
        elif nose_ratio > (0.5 + HEAD_POSE_THRESHOLD):
            return 'right'
        else:
            return 'center'
    
    def verify_challenge(self, landmarks):
        """
        Verify if user completed the current challenge.
        Returns (success: bool, message: str)
        """
        if not self.challenge_active or self.challenge_passed:
            return False, ""
        
        if self.failed_attempts >= 5:
            return False, "Maximum failed attempts reached. Please contact administrator."
        
        success = False
        message = ""
        
        # Check timeout
        if self.check_timeout():
            self.challenge_active = False
            success = False
            message = f"Timeout! (Challenge: {self.current_challenge})"
        
        # Verify based on challenge type
        elif self.current_challenge == 'blink':
            if self.detect_blink(landmarks):
                self.challenge_passed = True
                self.challenge_active = False
                success = True
                message = f"Blink Verified! (Count: {self.blink_counter})"
            else:
                assert self.challenge_start_time is not None
                elapsed = int(time.time() - self.challenge_start_time)
                success = False
                message = f"Blink ({elapsed}s)"
        
        elif self.current_challenge == 'look_left':
            pose = self.detect_head_pose(landmarks)
            if pose == 'left':
                self.challenge_passed = True
                self.challenge_active = False
                success = True
                message = "Look Left Verified!"
            else:
                success = False
                message = "Look Left"
        
        elif self.current_challenge == 'look_right':
            pose = self.detect_head_pose(landmarks)
            if pose == 'right':
                self.challenge_passed = True
                self.challenge_active = False
                success = True
                message = "Look Right Verified!"
            else:
                success = False
                message = "Look Right"
        
        else:
            success = False
            message = "Unknown Challenge"
        
        if success:
            self.failed_attempts = 0
        else:
            self.failed_attempts += 1
        
        return success, message
    
    def reset(self):
        """Reset liveness detector state"""
        self.challenge_active = False
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_passed = False
        self.blink_history.clear()
        self.blink_counter = 0
        self.failed_attempts = 0