
import os
import time
import traceback
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from pydub import AudioSegment
from datetime import datetime
import wave
import traceback

# Phase modules
import phase1
import phase2
import phase3

# Utilities
from utils.audio_storage import (
    generate_session_id, get_local_timestamp,
    build_filename, clear_directory
)
from dbsetup import (
    init_db, log_session, get_recent_sessions
)

# Flask setup
app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
SAMPLE_RATE = 16000

# Initialize database and folders
init_db()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)



def validate_audio_file(filepath):
    try:
        if not os.path.exists(filepath):
            raise ValueError("File does not exist")
        if os.path.getsize(filepath) == 0:
            raise ValueError("File is empty")


        audio = AudioSegment.from_file(filepath)
        if len(audio) == 0:
            raise ValueError("Audio contains no data")

        return True
    except Exception as e:
        raise Exception(f"Audio validation failed: {str(e)}")


def cleanup_files(*paths):
    """Delete specified files if they exist."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"[CLEANUP] Removed: {path}")
        except Exception as e:
            print(f"[WARN] Could not delete {path}: {e}")

def convert_to_wav(audio_file, output_path):
    """Convert any audio stream to WAV."""
    try:
        audio = AudioSegment.from_file(audio_file)
        audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}")

class PhaseProcessingError(Exception): pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    session_id = request.form.get("session_id") or generate_session_id()
    timestamp = get_local_timestamp()

    # Bypass toggles
    bypass = {
        'phase1': request.form.get("bypass_phase1", "false") == "true",
        'phase2': request.form.get("bypass_phase2", "false") == "true",
        'phase3': request.form.get("bypass_phase3", "false") == "true"
    }

    raw_path = os.path.join(UPLOAD_FOLDER, f"raw_{session_id}.wav")
    paths = {
        'preprocessed': os.path.join(PROCESSED_FOLDER, f"pre_{session_id}.wav"),
        'encoded': os.path.join(PROCESSED_FOLDER, f"enc_{session_id}.ogg"),
        'postprocessed': os.path.join(PROCESSED_FOLDER, f"post_{session_id}.wav")
    }

    try:
        # Robust WAV conversion: only keep true PCM WAV or re-encode
        try:
            header = file.read(4)
            file.seek(0)
            if file.filename.lower().endswith('.wav') and header == b'RIFF':
                file.save(raw_path)
            else:
                audio = AudioSegment.from_file(file)
                audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
                audio.export(raw_path, format='wav')
                print(f"[UPLOAD] Converted to WAV: {raw_path}")
        except Exception as conv_err:
            return jsonify({
                'error': 'File conversion failed',
                'details': str(conv_err),
                'supported_formats': ['wav', 'mp3', 'ogg', 'm4a', 'webm']
            }), 400

        validate_audio_file(raw_path)
        log_session(session_id, os.path.basename(raw_path), "baseline")

        # PHASE 1: Preprocessing
        if not bypass['phase1']:
            print(f"[PHASE 1] Starting preprocessing...")
            phase1.preprocess_audio(raw_path, paths['preprocessed'])
        else:
            print("[PHASE 1] Bypassed")
            paths['preprocessed'] = raw_path

        log_session(session_id, os.path.basename(paths['preprocessed']), "preprocessed")

        # PHASE 2: Encoding
        if not bypass['phase2']:
            print(f"[PHASE 2] Starting Opus encoding...")
            phase2.encode_to_opus(paths['preprocessed'], paths['encoded'], sample_rate=48000, bitrate='64k')
        else:
            print("[PHASE 2] Bypassed")
            paths['encoded'] = paths['preprocessed']

        log_session(session_id, os.path.basename(paths['encoded']), "encoded")

        print(f"[DEBUG] Phase 3 Input: {paths['encoded']}")
        print(f"[DEBUG] Phase 3 Output: {paths['postprocessed']}")

        # PHASE 3: Postprocessing
        if not bypass['phase3']:
            print(f"[PHASE 3] Starting postprocessing...")
            phase3.postprocess_audio(paths['encoded'], paths['postprocessed'],
                                     apply_denoise=True, apply_drc=True, apply_echo=False)
        else:
            print("[PHASE 3] Bypassed")
            paths['postprocessed'] = paths['encoded']

        log_session(session_id, os.path.basename(paths['postprocessed']), "postprocessed")

        if not os.path.exists(paths['postprocessed']):
            raise PhaseProcessingError("Final audio not created.")

        return jsonify({
            'status': 'complete',
            'session_id': session_id,
            'timestamp': timestamp,
            'raw_audio': f"/{raw_path}",
            'processed_audio': f"/{paths['postprocessed']}"
        })

    except Exception as e:
        cleanup_files(raw_path, *paths.values())
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500



@app.route('/submit_mos', methods=['POST'])
def submit_mos():
    data = request.get_json()
    try:
        ratings = {
            'clarity': int(data.get('clarity', 0)),
            'noise_reduction': int(data.get('noise_reduction', 0)),
            'level_matching': int(data.get('level_matching', 0))
        }
        if not all(1 <= v <= 5 for v in ratings.values()):
            return jsonify({'error': 'Each rating must be between 1 and 5'}), 400
        mos_avg = sum(ratings.values()) / 3.0
        from dbsetup import insert_mos_survey
        insert_mos_survey(
            clarity=ratings['clarity'],
            noise_reduction=ratings['noise_reduction'],
            level_matching=ratings['level_matching'],
            audio_file=data.get('audio_file', ''),
            mos_average=mos_avg
        )
        return jsonify({'status': 'MOS saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        clear_directory(UPLOAD_FOLDER, ext_filter=".wav")
        clear_directory(PROCESSED_FOLDER, ext_filter=".wav")
        return jsonify({'status': 'cleanup complete'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<path:filename>')
def serve_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
