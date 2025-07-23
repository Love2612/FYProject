
import os
import time
from datetime import datetime
import pytz
import uuid

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "static/processed"

# Nigeria time zone
NIGERIA_TZ = pytz.timezone("Africa/Lagos")

def get_local_timestamp():
    """Return timestamp adjusted to Nigeria local time."""
    return datetime.now(NIGERIA_TZ).strftime("%Y%m%d_%H%M%S")

def generate_session_id():
    """Generate unique session ID."""
    return f"SID_{uuid.uuid4().hex[:8]}"

def build_filename(base="audio", ext=".wav", folder=PROCESSED_DIR):
    """Create unique filename with timestamp."""
    timestamp = get_local_timestamp()
    filename = f"{base}_{timestamp}{ext}"
    path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    return path

def clear_directory(folder, ext_filter=None):
    """Clear files in a folder for testing/debugging."""
    if not os.path.exists(folder):
        return
    for file in os.listdir(folder):
        if ext_filter and not file.endswith(ext_filter):
            continue
        try:
            os.remove(os.path.join(folder, file))
        except:
            pass
