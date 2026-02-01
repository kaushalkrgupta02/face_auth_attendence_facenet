import pandas as pd
from datetime import datetime, timedelta
import os
from .config import LOG_PATH, COOLDOWN_SECONDS

class AttendanceManager:
    def __init__(self):
        self.user_state = {} # {Name: "IN" or "OUT"}
        self.last_action_time = {} # {Name: datetime object}
        self.load_logs()

    def load_logs(self):
        # Load state from existing CSV to prevent reset on restart
        if os.path.exists(LOG_PATH):
            try:
                df = pd.read_csv(LOG_PATH)
                if not df.empty:
                    # Replay logs to find current state
                    for index, row in df.iterrows():
                        name = row['Name']
                        action = row['Action']
                        self.user_state[name] = "IN" if action == "PUNCH IN" else "OUT"
                        
                        # Fix: Also load the time so we don't crash
                        # We use a dummy time (yesterday) so they aren't stuck in cooldown on restart
                        self.last_action_time[name] = datetime.now() - timedelta(days=1)
            except Exception as e:
                print(f"Error loading logs: {e}")

    def process_punch(self, name):
        now = datetime.now()
        
        # 1. SAFETY CHECK (The Fix for KeyError)
        # If user exists in CSV but not in memory (or missing timestamp), initialize them
        if name not in self.last_action_time:
            self.last_action_time[name] = now - timedelta(days=365) # Ready immediately
        if name not in self.user_state:
            self.user_state[name] = "OUT" # Default state

        # 2. Check Cooldown (Debounce)
        last_time = self.last_action_time[name]
        if (now - last_time).total_seconds() < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - (now - last_time).total_seconds())
            return "Wait", f"Wait {remaining}s"

        # 3. Determine Action
        current_state = self.user_state[name]
        action = "PUNCH IN" if current_state == "OUT" else "PUNCH OUT"
        
        # 4. Update State
        self.user_state[name] = "IN" if action == "PUNCH IN" else "OUT"
        self.last_action_time[name] = now
        
        # 5. Log to CSV
        self.log_to_csv(name, now, action)
        
        return "Success", action

    def log_to_csv(self, name, time, action):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame([[name, timestamp, action]], columns=['Name', 'Time', 'Action'])
        
        header = not os.path.exists(LOG_PATH)
        # Append to file
        new_row.to_csv(LOG_PATH, mode='a', header=header, index=False)
        print(f"Logged: {name} - {action}")