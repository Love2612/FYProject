import sqlite3
import os
from datetime import datetime

# Database configuration
DB_PATH = "audio_optimizer.db"
MAX_DB_SIZE_MB = 100  # Auto-cleanup threshold

def get_db_connection():
    """Enhanced DB connection with error handling and optimizations."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
    conn.execute("PRAGMA busy_timeout = 30000")  # 30-second timeout
    return conn

def init_database():
    """Initialize database with complete schema for all phases."""
    with get_db_connection() as conn:
        # MOS Surveys Table (Phase 4)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mos_surveys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES audio_metadata(session_id),
            clarity INTEGER NOT NULL CHECK (clarity BETWEEN 1 AND 5),
            noise_reduction INTEGER NOT NULL CHECK (noise_reduction BETWEEN 1 AND 5),
            level_matching INTEGER NOT NULL CHECK (level_matching BETWEEN 1 AND 5),
            mos_average REAL GENERATED ALWAYS AS (
                (clarity + noise_reduction + level_matching) / 3.0
            ) VIRTUAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES audio_metadata(session_id)
        )""")

        # Audio Metadata Table (Phases 1-4)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS audio_metadata (
            session_id TEXT PRIMARY KEY,
            raw_audio_path TEXT NOT NULL,
            processed_audio_path TEXT NOT NULL,
            rtt REAL,
            processing_time REAL,  
            phase1_time REAL,     
            phase2_time REAL,     
            phase3_time REAL,     
            snr_db REAL,          
            peaq_score REAL,      
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # Create indexes
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metadata_timestamp 
        ON audio_metadata(timestamp)""")
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mos_session 
        ON mos_surveys(session_id)""")

        print(f"Database initialized with schema v2 at {DB_PATH}")

def reset_database():
    """Safely reset database with confirmation."""
    if os.path.exists(DB_PATH):
        confirmation = input(f"Delete {DB_PATH}? (y/n): ")
        if confirmation.lower() == 'y':
            os.remove(DB_PATH)
            print("Database reset.")
    init_database()

def insert_mos_survey(session_id, clarity, noise_reduction, level_matching, audio_file=None):
    """Enhanced MOS survey insertion with session linking."""
    with get_db_connection() as conn:
        conn.execute("""
        INSERT INTO mos_surveys (
            session_id, clarity, noise_reduction, level_matching, audio_file
        ) VALUES (?, ?, ?, ?, ?)
        """, (session_id, clarity, noise_reduction, level_matching, audio_file))
        print(f"MOS survey saved for session {session_id}")

def insert_audio_metadata(
    session_id, 
    raw_audio_path, 
    processed_audio_path, 
    rtt=0.0,
    processing_time=None,
    phase_times=None,
    phase_metrics=None
):
    """Complete metadata storage for all phases."""
    with get_db_connection() as conn:
        conn.execute("""
        INSERT INTO audio_metadata (
            session_id,
            raw_audio_path,
            processed_audio_path,
            rtt,
            processing_time,
            phase1_time,
            phase2_time,
            phase3_time,
            snr_db,
            peaq_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            raw_audio_path,
            processed_audio_path,
            rtt,
            processing_time,
            phase_times.get('phase1') if phase_times else None,
            phase_times.get('phase2') if phase_times else None,
            phase_times.get('phase3') if phase_times else None,
            phase_metrics.get('snr_db') if phase_metrics else None,
            phase_metrics.get('peaq_score') if phase_metrics else None
        ))
        print(f"Metadata stored for session {session_id}")

def enforce_db_size_limit():
    """Auto-cleanup when DB exceeds size limit."""
    if os.path.getsize(DB_PATH) > MAX_DB_SIZE_MB * 1024 * 1024:
        with get_db_connection() as conn:
            # Delete oldest 10% of records
            conn.execute("""
            DELETE FROM audio_metadata 
            WHERE session_id IN (
                SELECT session_id 
                FROM audio_metadata 
                ORDER BY timestamp ASC 
                LIMIT (SELECT COUNT(*) / 10 FROM audio_metadata)
            )""")
            print(f"DB size limit enforced: {MAX_DB_SIZE_MB}MB")

def get_phase_stats():
    """Retrieve performance metrics for all phases."""
    with get_db_connection() as conn:
        return conn.execute("""
        SELECT 
            AVG(processing_time) as avg_total_time,
            AVG(phase1_time) as avg_phase1,
            AVG(phase2_time) as avg_phase2,
            AVG(phase3_time) as avg_phase3,
            AVG(snr_db) as avg_snr,
            AVG(peaq_score) as avg_peaq
        FROM audio_metadata
        WHERE processing_time IS NOT NULL
        """).fetchone()

if __name__ == "__main__":
    # Initialize with test data
    init_database()
    
    # Example usage
    test_id = "test123"
    insert_audio_metadata(
        session_id=test_id,
        raw_audio_path="uploads/test.wav",
        processed_audio_path="processed/test.wav",
        rtt=45.2,
        processing_time=1.23,
        phase_times={'phase1': 0.4, 'phase2': 0.3, 'phase3': 0.5},
        phase_metrics={'snr_db': 15.2, 'peaq_score': 0.92}
    )
    
    insert_mos_survey(
        session_id=test_id,
        clarity=4,
        noise_reduction=5,
        level_matching=3
    )
    
    print("Phase stats:", dict(get_phase_stats()))