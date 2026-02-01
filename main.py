import cv2
import argparse
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

        # Draw UI
        detector.draw_roi(frame)
        
        # 1. Detect
        faces = detector.detect(frame)

        if isinstance(faces, dict):
            for key in faces:
                face_data = faces[key]
                box = face_data['facial_area']
                
                # 2. Check Intent
                valid_intent, msg = detector.verify_intent(face_data, frame.shape[1], frame.shape[0])
                
                color = (0, 0, 255) # Red by default
                
                if valid_intent:
                    color = (0, 255, 255) # Yellow (Processing)
                    
                    # 3. Recognize
                    emb = recognizer.get_embedding(frame, face_data)
                    if emb is not None:
                        name, dist = recognizer.identify(emb)
                        
                        if args.mode == 'register':
                            # Registration Mode
                            if args.name:
                                cv2.putText(frame, "Press 'S' to Save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                if cv2.waitKey(1) & 0xFF == ord('s'):
                                    recognizer.register_face(args.name, emb)
                                    print(f"Registered {args.name}")
                                    return # Exit after save
                            else:
                                cv2.putText(frame, "Error: Provide --name", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        elif args.mode == 'run':
                            # Attendance Mode
                            if name != "Unknown":
                                color = (0, 255, 0) # Green
                                status, action = manager.process_punch(name)
                                display_text = f"{name}: {action}" if status == "Success" else f"{name}: Wait..."
                                cv2.putText(frame, display_text, (box[0], box[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            else:
                                cv2.putText(frame, "Unknown", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw Face Box & Feedback
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