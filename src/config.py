import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "face_db.pt")
LOG_PATH = os.path.join(DATA_DIR, "attendance_log.csv")

# Create data dir if missing
os.makedirs(DATA_DIR, exist_ok=True)

# System Settings
DEVICE_ID = 0 # Camera Index
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection & Intent
RETINA_CONFIDENCE = 0.90
MIN_FACE_WIDTH = 80         # Pixels (Too far check)
ROI_CENTER_PCT = 0.40       # Center 40% box
GAZE_THRESHOLD_LOW = 0.35   # Head turn limit
GAZE_THRESHOLD_HIGH = 0.65  # Head turn limit

# Recognition
RECOGNITION_THRESHOLD = 0.60 # Lower = stricter

# Attendance Logic
COOLDOWN_SECONDS = 300      # 5 Minutes buffer