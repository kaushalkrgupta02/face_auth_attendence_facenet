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

# Liveness Detection (Spoof Prevention)
LIVENESS_ENABLED = True
LIVENESS_CHALLENGE_TIMEOUT = 5  # seconds to complete challenge
LIVENESS_CHALLENGES = ['blink', 'look_left', 'look_right']  # Available challenges

# Blink Detection
BLINK_THRESHOLD = 0.25  # Eye aspect ratio threshold
BLINK_FRAMES = 3  # Frames to confirm blink

# Head Pose Detection (for look left/right)
HEAD_POSE_THRESHOLD = 0.5  # Ratio threshold for left/right gaze

# Recognition
RECOGNITION_THRESHOLD = 0.60 # Lower = stricter

# Attendance Logic
COOLDOWN_SECONDS =  60    # 1 Minutes buffer for test you can put accordingly

# CSV Columns (Attendance Log Schema)
CSV_COLUMNS = ['Name', 'Date', 'Punch In Time', 'Punch Out Time']

# Table Configuration (for UI display)
TABLE_CONFIG = {
    "columns": [
        {
            "field": "Name",
            "title": "Name",
            "minWidth": 40,
            "resizable": True,
            "sorter": "string",
            "headerSort": True,
            "width": 93
        },
        {
            "field": "Date",
            "title": "Date",
            "minWidth": 40,
            "resizable": True,
            "sorter": "string",
            "headerSort": True,
            "width": 120
        },
        {
            "field": "Punch In Time",
            "title": "Punch In Time",
            "minWidth": 40,
            "resizable": True,
            "sorter": "string",
            "headerSort": True,
            "width": 140
        },
        {
            "field": "Punch Out Time",
            "title": "Punch Out Time",
            "minWidth": 40,
            "resizable": True,
            "sorter": "string",
            "headerSort": True,
            "width": 140
        }
    ]
}