import pandas as pd
from datetime import datetime, timedelta
import os
from .config import LOG_PATH, COOLDOWN_SECONDS

class AttendanceManager:
    def __init__(self):
        self.user_state = {} # {Name: "IN" or "OUT"}
        self.last_action_time = {} # {Name: datetime}
        self.load_logs()

    def load_logs(self):
        # Load state from existing CSV to prevent reset on restart
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            # Replay logs to find current state (Basic Implementation)
            for index, row in df.iterrows():
                self.user_state[row['Name']] = "IN" if row['Action'] == "PUNCH IN" else "OUT"

    def process_punch(self, name):
        now = datetime.now()
        
        # Initialize
        if name not in self.user_state:
            self.user_state[name] = "OUT"
            self.last_action_time[name] = now - timedelta(days=365) # Ready immediately

        # Check Cooldown (Debounce)
        last_time = self.last_action_time[name]
        if (now - last_time).total_seconds() < COOLDOWN_SECONDS:
            return "Wait", None

        # Determine Action
        current_state = self.user_state[name]
        action = "PUNCH IN" if current_state == "OUT" else "PUNCH OUT"
        
        # Update State
        self.user_state[name] = "IN" if action == "PUNCH IN" else "OUT"
        self.last_action_time[name] = now
        
        # Log to CSV
        self.log_to_csv(name, now, action)
        
        return "Success", action

    def log_to_csv(self, name, time, action):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame([[name, timestamp, action]], columns=['Name', 'Time', 'Action'])
        
        header = not os.path.exists(LOG_PATH)
        new_row.to_csv(LOG_PATH, mode='a', header=header, index=False)
        print(f"Logged: {name} - {action}")