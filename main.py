import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import argparse
import time
from datetime import datetime  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.attendance import AttendanceManager
from src.liveness import LivenessDetector
from core.config import FRAME_WIDTH, FRAME_HEIGHT, LIVENESS_ENABLED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['run', 'register'], required=True, help='Mode: run or register')
    parser.add_argument('--name', type=str, help='Name of user (required for register)')
    args = parser.parse_args()

    # Initialize Modules
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    manager = AttendanceManager()
    # liveness = LivenessDetector() if LIVENESS_ENABLED else None
    liveness = None

    # CHECK IF REGISTERING - VALIDATE NAME BEFORE STARTING
    if args.mode == 'register':
        if not args.name:
            print("Error: You must provide --name for registration.")
            return
        
        if recognizer.check_name_exists(args.name):
            print(f"\n[ERROR] User '{args.name}' is already registered!")
            print("[HINT] Please use a different name to register.\n")
            return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"Starting System in {args.mode.upper()} mode...")

    # pending_user = None  # User awaiting liveness verification

    while True:
        ret, frame = cap.read()
        if not ret: break

        detector.draw_roi(frame)
        faces = detector.detect(frame)

        if isinstance(faces, dict):
            for key in faces:
                face_data = faces[key]
                box = face_data['facial_area']
                
                valid_intent, msg = detector.verify_intent(face_data, frame.shape[1], frame.shape[0])
                
                color = (0, 0, 255)  # Red by default
                
                if valid_intent:
                    color = (0, 255, 255)  # Yellow (Processing)
                    
                    # --- MODE: REGISTER ---
                    if args.mode == 'register':
                        cv2.putText(frame, "Press 'S' to Capture (5 Samples)", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            print(f"Starting capture for {args.name}...")
                            samples = []
                            sample_count = 0
                            
                            while sample_count < 5:
                                ret, temp_frame = cap.read()
                                if not ret: break
                                
                                temp_faces = detector.detect(temp_frame)
                                if temp_faces:
                                    temp_data = list(temp_faces.values())[0]
                                    temp_box = temp_data['facial_area']
                                    
                                    cv2.rectangle(temp_frame, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), (0, 255, 0), 2)
                                    cv2.putText(temp_frame, f"Capturing {sample_count+1}/5", (50, 240), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    cv2.imshow('Face Attendance', temp_frame)
                                    cv2.waitKey(200) 
                                    
                                    emb = recognizer.get_embedding(temp_frame, temp_data)
                                    if emb is not None:
                                        samples.append(emb)
                                        sample_count += 1
                                        print(f"Captured {sample_count}/5")
                            
                            if recognizer.register_face(args.name, samples):
                                print(f"User {args.name} Registered Successfully!")
                                return 
                            else:
                                print(f"Registration failed for {args.name}")
                                return
                            
                    # --- MODE: RUN (Attendance with Liveness) ---
                    elif args.mode == 'run':
                        # Step 1: Initial face recognition
                        emb = recognizer.get_embedding(frame, face_data)
                        if emb is not None:
                            name, dist = recognizer.identify(emb)
                            
                            if name != "Unknown":
                                color = (0, 255, 0)  # Green
                                
                                # # Start liveness challenge
                                # if LIVENESS_ENABLED and liveness is not None:
                                #     pending_user = name
                                #     challenge = liveness.start_challenge()
                                #     print(f"[LIVENESS] Asking {name} to: {challenge}")
                                # else:
                                #     # Liveness disabled, proceed directly
                                status, action = manager.process_punch(name)
                                if status == "Success":
                                    time_str = datetime.now().strftime('%d-%b %H:%M:%S')
                                    display_text = f"{name}: {action}"
                                    sub_text = f"at {time_str}"
                                    cv2.putText(frame, display_text, (box[0], box[1]-30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    cv2.putText(frame, sub_text, (box[0], box[1]-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                                else:
                                    cv2.putText(frame, f"{name}: Wait...", (box[0], box[1]-20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            else:
                                cv2.putText(frame, "Unknown", (box[0], box[1]-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        # # Step 2: Liveness verification
                        # elif pending_user is not None and LIVENESS_ENABLED and liveness is not None:
                            
                        #     liveness.draw_eye_landmarks(frame, face_data['landmarks'])
                        #     liveness_verified, liveness_msg = liveness.verify_challenge(face_data['landmarks'])

                        #     if liveness_verified:
                        #         # Liveness passed, now do punch
                        #         color = (0, 255, 0)  # Green
                        #         status, action = manager.process_punch(pending_user)
                                
                        #         if status == "Success":
                        #             time_str = datetime.now().strftime('%d-%b %H:%M:%S')
                        #             display_text = f"{pending_user}: {action}"
                        #             sub_text = f"at {time_str}"
                        #             cv2.putText(frame, display_text, (box[0], box[1]-30), 
                        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        #             cv2.putText(frame, sub_text, (box[0], box[1]-10), 
                        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                        #             cv2.putText(frame, liveness_msg, (box[0], box[1]+20), 
                        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        #             pending_user = None
                        #             liveness.reset()
                        #     elif liveness.check_timeout():
                        #         # Liveness failed - timeout
                        #         color = (0, 0, 255)  # Red
                        #         cv2.putText(frame, f"{liveness_msg}", (box[0], box[1]-30), 
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        #         pending_user = None
                        #         liveness.reset()
                        #     else:
                        #         # Liveness in progress
                        #         color = (0, 165, 255)  # Orange
                        #         cv2.putText(frame, f"Liveness: {liveness_msg}", (box[0], box[1]-30), 
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw Face Box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                if not valid_intent:
                    cv2.putText(frame, msg, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Face Attendance', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()