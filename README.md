# Face Authentication Attendance System

A real-time biometric attendance system using **FaceNet** for face recognition and **RetinaFace** for robust face detection.

## System Overview

The system has three core functionalities:

1. **Register User's Face** - Captures 5 face samples and stores embeddings
2. **Identify Face** - Matches detected faces against registered database
3. **Mark Attendance** - Records punch-in/punch-out with automatic state management

## Technical Approach

### Model and Approach Used
I utilize a **cascaded pipeline** where **RetinaFace (ResNet50)** performs dense face localisation  to handle varied lighting/poses, followed by **FaceNet (InceptionResnetV1)** which maps aligned faces into a 512-dimensional Euclidean space for identity verification.

### Training Process
I employ **One-Shot Learning** using pre-trained weights (VGGFace2). Instead of training the model from scratch, we perform a **registration phase**  where 5 burst samples are captured, aligned, and averaged to create a robust prototype embedding for each user.

### Face Detection: RetinaFace
- **Model**: ResNet50-based RetinaFace
- **Confidence Threshold**: 0.90 (strict detection)
- **Why**: Handles varying lighting, angles, and partially occluded faces
- **Performance**: Real-time detection at 640×480 resolution

### Face Recognition: FaceNet (InceptionResNetV1)
- **Model**: InceptionResNetV1 (pre-trained on VGGFace2)
- **Embedding Dimension**: 128D vector space
- **Distance Metric**: Euclidean distance
- **Recognition Threshold**: 0.60 (configurable)
- **Why**: Pre-trained model provides high accuracy without requiring custom training data

### Implementation for Spoof Prevention 
Prevents accidental or fraudulent punches through multi-factor checks:

1. **ROI (Region of Interest)**
   - Face must be centered in frame (within 40% center area)
   - Filters out passersby and side-angle attempts

2. **Minimum Face Size**
   - Face width must be ≥ 80 pixels
   - Prevents spoofing with printed photos held far away

3. **Head Pose Analysis**
   - Nose position relative to eyes determines head direction
   - Threshold: 0.5 ratio (left/right/center detection)
   - Ensures person is looking at camera

4. **Liveness Detection** (testing)
   - Eye blink detection using Eye Aspect Ratio (EAR)
   - Head movement challenges (look left/right)
   - Maximum 5 failed attempts before blocking user (then go to admin for attendence)

### Attendance Logic
- **State Machine**: Records punch-in, then automatically switches to punch-out mode
- **Cooldown**: 300-second buffer prevents duplicate punches
- **Logging**: CSV format with timestamp and punch type (In future db can be connected [code modularity])

## Installation

### 1. Clone repository
git clone <repo-url>
cd face_auth_attendence_facenet

### 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Verify camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera failed')"

### 5a. Registration of new user
python main.py --mode register --name "Alice"
(Note that name should be unique)

### 5b. Ready to run and MARK ATTENDANCE
python main.py --mode run



## Accuracy & Performance

### Expected Recognition Accuracy

| Condition | Accuracy | Notes |
|---------|---------|------|
| Frontal face, good lighting | 96–98% | Ideal conditions |
| Slight angle (~20°), good lighting | 93–95% | Realistic daily use |
| Moderate angle (~30°), average lighting | 90–93% | Acceptable performance |
| Large angle (45°+) | 75–85% | Intent verification filters applied |
| Poor lighting (<100 lux) | 80–90% | Degraded but functional |
| Extreme conditions (low light + large angle) | 60–75% | Higher risk of false rejection |


## Known Limitations

Reduced accuracy for identical twins

Occluded faces (masks, caps) may fail detection

No protection against high-quality video replay attacks

Not suitable for night-time outdoor use without IR cameras

Somtimes Liveness failed for blink task if asked to users


## Conclusion

This project demonstrates a practical face authentication–based attendance system using modern deep learning models.
It balances accuracy, real-time performance, and implementation simplicity while acknowledging the inherent limitations of face recognition systems.



### Made With FaceNet + RetinaFace 
### Author @kaushalkrgupta02