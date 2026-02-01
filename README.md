# Face Authentication Attendance System

A high-accuracy biometric attendance system utilizing **RetinaFace** for robust detection and **FaceNet** (Inception Resnet V1) for recognition. Features "Intent Verification" to prevent accidental punches and attempted spoofing.

## Features
* **Robust Detection:** Uses RetinaFace (ResNet50) to handle low light and side angles.
* **Intent Verification:** Filters passersby using ROI, Minimum Size, and Head Pose analysis.
* **Logic Engine:** Implements a state machine (In/Out) with a 5-minute debounce cooldown.
* **Local Database:** Portable `.pt` embedding storage and `.csv` logging.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt