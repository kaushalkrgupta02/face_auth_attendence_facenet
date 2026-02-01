import pandas as pd
from datetime import datetime, timedelta
import os
from core.config import LOG_PATH, COOLDOWN_SECONDS, CSV_COLUMNS

class AttendanceManager:
    def __init__(self):
        self.user_state = {} # {Name: "IN" or "OUT"}
        self.last_action_time = {} # {Name: datetime object}
        self.load_logs()

    def load_logs(self):
        if os.path.exists(LOG_PATH):
            try:
                df = pd.read_csv(LOG_PATH)
                if not df.empty:
                    # Get the last entry for each user to restore their state
                    for name in df['Name'].unique():
                        user_df = df[df['Name'] == name]
                        last_row = user_df.iloc[-1]
                        
                        # If punch out time is NaN, user is still IN
                        punch_out = last_row['Punch Out Time']
                        self.user_state[name] = "OUT" if (pd.notna(punch_out) and str(punch_out).strip()) else "IN"
                        
                        # Set cooldown time
                        self.last_action_time[name] = datetime.now() - timedelta(days=1)
            except Exception as e:
                print(f"Error loading logs: {e}")

    def process_punch(self, name):
        now = datetime.now()
        
        # Initialize user if not seen before
        if name not in self.last_action_time:
            self.last_action_time[name] = now - timedelta(days=365)
        if name not in self.user_state:
            self.user_state[name] = "OUT"

        # Check Cooldown (Debounce)
        last_time = self.last_action_time[name]
        if (now - last_time).total_seconds() < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - (now - last_time).total_seconds())
            return "Wait", f"Wait {remaining}s"

        # Determine Action based on current state
        action = "PUNCH IN" if self.user_state[name] == "OUT" else "PUNCH OUT"
        
        # Update State and timestamp
        self.user_state[name] = "IN" if action == "PUNCH IN" else "OUT"
        self.last_action_time[name] = now
        
        # Log to CSV
        self.log_to_csv(name, now, action)
        
        return "Success", action

    def log_to_csv(self, name, time, action):
        today_date = time.strftime('%Y-%m-%d')
        time_str = time.strftime('%H:%M:%S')
        
        # Ensure CSV file exists with headers
        if not os.path.exists(LOG_PATH):
            df = pd.DataFrame(columns=CSV_COLUMNS)
            # Set all columns as string type
            for col in CSV_COLUMNS:
                df[col] = df[col].astype(str)
            df.to_csv(LOG_PATH, index=False)
        
        # Load existing data and ensure proper dtype
        df = pd.read_csv(LOG_PATH, dtype=str)
        
        # Find today's entry for this user
        today_mask = (df['Name'] == name) & (df['Date'] == today_date)
        
        if action == "PUNCH IN":
            if today_mask.any():
                df.loc[today_mask, 'Punch In Time'] = time_str
            else:
                new_row = pd.DataFrame([[name, today_date, time_str, '']], columns=CSV_COLUMNS)
                df = pd.concat([df, new_row], ignore_index=True)
        else:  # PUNCH OUT
            if today_mask.any():
                df.loc[today_mask, 'Punch Out Time'] = time_str
            else:
                new_row = pd.DataFrame([[name, today_date, '', time_str]], columns=CSV_COLUMNS)
                df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_csv(LOG_PATH, index=False)
        print(f"Logged: {name} - {action} at {time_str}")