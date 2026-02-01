import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import argparse
import time

import cv2
import argparse
import time
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.attendance import AttendanceManager
from src.config import FRAME_WIDTH, FRAME_HEIGHT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['run', 'register'], required=True, help='Mode: run or register')
    parser.add_argument('--name', type=str, help='Name of user (required for register)')
    args = parser.parse_args()

    # Initialize Modules
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    manager = AttendanceManager()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"Starting System in {args.mode.upper()} mode...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Draw ROI (White Box)
        detector.draw_roi(frame)
        
        # 1. Detect
        faces = detector.detect(frame)

        if isinstance(faces, dict):
            for key in faces:
                face_data = faces[key]
                box = face_data['facial_area']
                
                # 2. Check Intent (Are they looking at camera?)
                valid_intent, msg = detector.verify_intent(face_data, frame.shape[1], frame.shape[0])
                
                color = (0, 0, 255) # Red by default
                
                if valid_intent:
                    color = (0, 255, 255) # Yellow (Processing)
                    
                    # --- MODE: REGISTER ---
                    if args.mode == 'register':
                        cv2.putText(frame, "Press 'S' to Capture (5 Samples)", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Trigger Capture
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            if not args.name:
                                print("Error: You must provide --name for registration.")
                                return

                            print(f"Starting capture for {args.name}...")
                            samples = []
                            sample_count = 0
                            
                            # BURST MODE LOOP
                            while sample_count < 5:
                                ret, temp_frame = cap.read()
                                if not ret: break
                                
                                # Re-detect in new frame
                                temp_faces = detector.detect(temp_frame)
                                if temp_faces:
                                    # Get largest face
                                    temp_data = list(temp_faces.values())[0]
                                    temp_box = temp_data['facial_area']
                                    
                                    # Draw feedback
                                    cv2.rectangle(temp_frame, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), (0, 255, 0), 2)
                                    cv2.putText(temp_frame, f"Capturing {sample_count+1}/5", (50, 240), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    cv2.imshow('Face Attendance', temp_frame)
                                    cv2.waitKey(200) # Wait 200ms
                                    
                                    # Get Embedding
                                    emb = recognizer.get_embedding(temp_frame, temp_data)
                                    if emb is not None:
                                        samples.append(emb)
                                        sample_count += 1
                                        print(f"Captured {sample_count}/5")
                            
                            # Save the accumulated samples
                            if recognizer.register_face(args.name, samples):
                                print(f"User {args.name} Registered Successfully!")
                                return # Exit program after success

                    # --- MODE: RUN (Attendance) ---
                    elif args.mode == 'run':
                        emb = recognizer.get_embedding(frame, face_data)
                        if emb is not None:
                            name, dist = recognizer.identify(emb)
                            
                            if name != "Unknown":
                                color = (0, 255, 0) # Green
                                status, action = manager.process_punch(name)
                                display_text = f"{name}: {action}" if status == "Success" else f"{name}: Wait..."
                                cv2.putText(frame, display_text, (box[0], box[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            else:
                                cv2.putText(frame, "Unknown", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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