

import sqlite3
import os
from datetime import datetime

DB_PATH = "audio_optimizer.db"

def init_db():
    """Initialize the database if it doesn't exist."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            file_type TEXT,  
            timestamp TEXT,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

# -- "baseline" or "processed"

def log_session(session_id, filename, file_type="processed", notes=None):
    """Log audio session info into database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sessions (session_id, filename, file_type, timestamp, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, filename, file_type, timestamp, notes))
    conn.commit()
    conn.close()
    print(f"[DB] Logged session: {session_id} | {file_type} | {filename}")

def get_recent_sessions(limit=10):
    """Fetch latest audio sessions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT session_id, filename, file_type, timestamp, notes
        FROM sessions
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Run once at server startup
if __name__ == "__main__":
    init_db()
    print("[DB] Initialized.")
