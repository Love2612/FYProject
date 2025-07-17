import sqlite3
import os
import uuid
from datetime import datetime, timedelta

# Configuration
DB_PATH = "audio_optimizer.db"
UPLOADS_DIR = "uploads"  # Lowercase for consistency
PROCESSED_DIR = "static/processed"
CLEANUP_THRESHOLD_HOURS = 24
MAX_DB_ENTRIES = 1000  # New: Prevent DB bloating

def get_db_connection():
    """Enhanced DB connection with timeout and busy handler."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")  # 30-second timeout
    return conn

def ensure_directories():
    """Create required directories with permission checks."""
    for directory in [UPLOADS_DIR, PROCESSED_DIR]:
        try:
            os.makedirs(directory, exist_ok=True, mode=0o755)
        except OSError as e:
            raise RuntimeError(f"Failed to create directory {directory}: {str(e)}")

def generate_session_id():
    """Generate URL-safe session ID."""
    return str(uuid.uuid4()).replace('-', '')[:16]  # Shorter ID for URLs

def store_audio_metadata(session_id, raw_audio_path, processed_audio_path, rtt, 
                        processing_time=None, phase_metrics=None):
    """
    Enhanced metadata storage with phase metrics.
    
    Args:
        phase_metrics: dict containing snr_db, peaq_score from phase4
        processing_time: float in seconds
    """
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO audio_metadata (
                session_id, 
                raw_audio_path, 
                processed_audio_path, 
                rtt,
                processing_time,
                snr_db,
                peaq_score,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            session_id,
            raw_audio_path,
            processed_audio_path,
            rtt,
            processing_time,
            phase_metrics.get('snr_db', None) if phase_metrics else None,
            phase_metrics.get('peaq_score', None) if phase_metrics else None
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise ValueError(f"Duplicate session ID: {session_id}")
    finally:
        conn.close()

def get_audio_metadata(session_id):
    """Get metadata with additional phase metrics."""
    conn = get_db_connection()
    try:
        row = conn.execute("""
            SELECT *, 
                   datetime(timestamp, 'localtime') as local_timestamp
            FROM audio_metadata 
            WHERE session_id = ?
        """, (session_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_recent_metadata(limit=50):
    """Get recent entries with pagination."""
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT *, 
                   datetime(timestamp, 'localtime') as local_timestamp
            FROM audio_metadata 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def cleanup_old_audio(age_hours=CLEANUP_THRESHOLD_HOURS, max_entries=MAX_DB_ENTRIES):
    """Enhanced cleanup with:
    1. Age-based file deletion
    2. DB entry limit enforcement
    3. Transaction safety
    """
    conn = get_db_connection()
    try:
        # Age-based cleanup
        threshold = datetime.now() - timedelta(hours=age_hours)
        old_entries = conn.execute("""
            SELECT session_id, raw_audio_path, processed_audio_path 
            FROM audio_metadata 
            WHERE timestamp < ?
        """, (threshold,)).fetchall()

        deleted_files = 0
        for entry in old_entries:
            try:
                for path in [entry['raw_audio_path'], entry['processed_audio_path']]:
                    if path and os.path.exists(path):
                        os.remove(path)
                        deleted_files += 1
            except OSError:
                continue

        conn.execute("DELETE FROM audio_metadata WHERE timestamp < ?", (threshold,))
        
        # Count-based cleanup if DB is too large
        count = conn.execute("SELECT COUNT(*) FROM audio_metadata").fetchone()[0]
        if count > max_entries:
            excess = count - max_entries
            oldest = conn.execute("""
                SELECT session_id, raw_audio_path, processed_audio_path
                FROM audio_metadata
                ORDER BY timestamp ASC
                LIMIT ?
            """, (excess,)).fetchall()
            
            for entry in oldest:
                try:
                    for path in [entry['raw_audio_path'], entry['processed_audio_path']]:
                        if path and os.path.exists(path):
                            os.remove(path)
                except OSError:
                    pass
            
            conn.execute("""
                DELETE FROM audio_metadata
                WHERE session_id IN (
                    SELECT session_id
                    FROM audio_metadata
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            """, (excess,))
        
        conn.commit()
        return {
            "age_deleted": len(old_entries),
            "files_deleted": deleted_files,
            "total_remaining": conn.execute("SELECT COUNT(*) FROM audio_metadata").fetchone()[0]
        }
    finally:
        conn.close()

def init_database():
    """Initialize DB with schema supporting phase metrics."""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_metadata (
                session_id TEXT PRIMARY KEY,
                raw_audio_path TEXT NOT NULL,
                processed_audio_path TEXT NOT NULL,
                rtt REAL,
                processing_time REAL,
                snr_db REAL,
                peaq_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON audio_metadata (timestamp)
        """)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    # Test database initialization
    init_database()
    ensure_directories()
    
    # Test metadata storage
    test_id = generate_session_id()
    store_audio_metadata(
        session_id=test_id,
        raw_audio_path=f"uploads/test_{test_id}.wav",
        processed_audio_path=f"static/processed/processed_{test_id}.wav",
        rtt=42.5,
        processing_time=1.23,
        phase_metrics={"snr_db": 15.2, "peaq_score": 0.92}
    )
    
    # Verify retrieval
    print("Test metadata:", get_audio_metadata(test_id))
    
    # Test cleanup
    print("Cleanup results:", cleanup_old_audio(age_hours=0.01))  # 36s threshold for testing