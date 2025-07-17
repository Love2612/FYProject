import os
import time
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from pydub import AudioSegment
import phase1
import phase2
import phase3
import phase4
from dbsetup import init_database
from utils.audio_storage import ensure_directories, generate_session_id, store_audio_metadata, cleanup_old_audio
from datetime import datetime
import traceback
import wave  # For additional audio validation if needed
import mimetypes

app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app)

# Configuration
UPLOAD_FOLDER = 'uploads'  # Changed to lowercase for consistency
PROCESSED_FOLDER = 'static/processed'
SAMPLE_RATE = 16000  # Updated from 48000 to match all phases
MAX_RECORDING_DURATION = 4 * 60  # 4 minutes in seconds

# Initialize database and directories
init_database()
ensure_directories()

# Add these utility functions right after imports, before route definitions
def validate_audio_file(filepath):
    """Verify the audio file is valid and readable."""
    try:
        if not os.path.exists(filepath):
            raise ValueError("File does not exist")
        
        if os.path.getsize(filepath) == 0:
            raise ValueError("File is empty")
        
        try:
            audio = AudioSegment.from_file(filepath)
            if len(audio) == 0:
                raise ValueError("Audio contains no data")
            return True
        except Exception as e:
            raise ValueError(f"Invalid audio file: {str(e)}")
    except Exception as e:
        raise Exception(f"Audio validation failed: {str(e)}")

def cleanup_files(*paths):
    """Clean up temporary files."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Cleaned up: {path}")
            except Exception as e:
                print(f"Warning: Could not delete {path}: {str(e)}")
                
def convert_to_wav(audio_data, output_path):
    """Convert any audio data to WAV format using pydub"""
    try:
        # Create AudioSegment from raw data
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        # Export as WAV
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Conversion error: {str(e)}")
        return False

def is_valid_wav(filepath):
    """Check if file is a valid WAV file"""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            return wav_file.getnchannels() > 0
    except:
        return False

# Custom Exceptions
class PhaseProcessingError(Exception):
    """Base class for phase-specific errors"""
    pass

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_audio():
#     """Handle file upload and processing pipeline."""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     # Generate session and file paths
#     session_id = generate_session_id()
#     raw_filename = f'raw_{session_id}.wav'
#     raw_path = os.path.join(UPLOAD_FOLDER, raw_filename)
#     file.save(raw_path)
    
#     try:
#         # Phase Processing Pipeline
#         start_time = time.time()
#         paths = {
#             'preprocessed': os.path.join(PROCESSED_FOLDER, f'preprocessed_{session_id}.wav'),
#             'encoded': os.path.join(PROCESSED_FOLDER, f'encoded_{session_id}.opus'),
#             'postprocessed': os.path.join(PROCESSED_FOLDER, f'postprocessed_{session_id}.wav'),
#             'results': os.path.join(PROCESSED_FOLDER, f'evaluation_{session_id}.json')
#         }

#         # Execute phases with error wrapping
#         try:
#             phase1.preprocess_audio(raw_path, paths['preprocessed'])
#         except Exception as e:
#             raise PhaseProcessingError(f'Phase 1 (Preprocessing) failed: {str(e)}')

#         try:
#             phase2.encode_to_opus(paths['preprocessed'], paths['encoded'])
#         except Exception as e:
#             raise PhaseProcessingError(f'Phase 2 (Encoding) failed: {str(e)}')

#         try:
#             phase3.postprocess_audio(paths['encoded'], paths['postprocessed'])
#         except Exception as e:
#             raise PhaseProcessingError(f'Phase 3 (Postprocessing) failed: {str(e)}')

#         end_time = time.time()
        
#         # Evaluation Phase
#         webrtc_rtt = float(request.form.get('rtt', 0.0))
#         results = phase4.evaluate_audio(
#             raw_path, 
#             paths['postprocessed'], 
#             paths['results'],
#             [start_time, end_time],
#             webrtc_rtt
#         )

#         # Enhanced Metadata Storage
#         store_audio_metadata(
#             session_id=session_id,
#             raw_path=raw_path,
#             processed_path=paths['postprocessed'],
#             webrtc_rtt=webrtc_rtt,
#             processing_time=end_time - start_time,
#             phase_metrics={
#                 'snr_db': results.get('snr_db', 0),
#                 'peaq_score': results.get('peaq_score', 0)
#             }
#         )

#         return jsonify({
#             'status': 'Processing complete',
#             'session_id': session_id,
#             'raw_audio': f'/{raw_path}',
#             'processed_audio': f'/{paths["postprocessed"]}',
#             'results': results
#         })

#     except PhaseProcessingError as e:
#         return jsonify({'error': str(e)}), 500
#     except Exception as e:
#         return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle file upload and processing pipeline with robust error handling."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'details': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'details': 'Empty filename'}), 400

    # Accept session ID from frontend if provided
    frontend_sid = request.form.get("session_id")
    session_id = frontend_sid if frontend_sid else generate_session_id()


    # Generate session and file paths
    raw_filename = f'raw_{session_id}.wav'
    raw_path = os.path.join(UPLOAD_FOLDER, raw_filename)
    paths = {
        'preprocessed': os.path.join(PROCESSED_FOLDER, f'preprocessed_{session_id}.wav'),
        'encoded': os.path.join(PROCESSED_FOLDER, f'encoded_{session_id}.opus'),
        'postprocessed': os.path.join(PROCESSED_FOLDER, f'postprocessed_{session_id}.wav'),
        'results': os.path.join(PROCESSED_FOLDER, f'evaluation_{session_id}.json')
    }
    
    try:
        # Save the file first with explicit WAV conversion if needed
        if not file.filename.lower().endswith('.wav'):
            # try:
            #     file_type = mimetypes.guess_type(file.filename)[0]
            #     if file_type == "audio/webm" or file.filename.lower().endswith(".webm"):
            #         audio = AudioSegment.from_file(file, format="webm")
            #     else:
            
            
                # audio = AudioSegment.from_file(file)
                # audio.export(raw_path, format="wav")
                
                
                
            # except Exception as conv_err:
            #     return jsonify({
            #         'error': 'File conversion failed',
            #         'details': str(conv_err),
            #         'supported_formats': ['wav', 'mp3', 'ogg', 'm4a']
            #     }), 400
            
            try:
                # Force WebM if suspected from header
                header = file.read(4)
                file.seek(0)

                if header == b'\x1aE\xdf\xa3':  # WebM magic bytes
                    audio = AudioSegment.from_file(file, format='webm')
                else:
                    try:
                        audio = AudioSegment.from_file(file)  # Let ffmpeg infer format
                    except Exception as err:
                        raise ValueError(f"Unsupported or corrupt audio file. Details: {err}")

                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(raw_path, format='wav')

            except Exception as conv_err:
                return jsonify({
                    'error': 'File conversion failed',
                    'details': str(conv_err),
                    'supported_formats': ['wav', 'mp3', 'ogg', 'm4a', 'webm']
                }), 400

        else:
            file.save(raw_path)

        # Validate the saved file
        try:
            validate_audio_file(raw_path)
        except Exception as validation_err:
            cleanup_files(raw_path)
            return jsonify({
                'error': 'Invalid audio file',
                'details': str(validation_err)
            }), 400

        # Phase Processing Pipeline
        start_time = time.time()
        
        try:
            # Phase 1: Preprocessing
            t1 = time.time()
            print(f"Starting Phase 1 preprocessing for {raw_path}")
            phase1.preprocess_audio(raw_path, paths['preprocessed'])
            print(f"Phase 1 completed in {time.time() - t1:.2f}s")

            # Phase 2: Encoding
            t1 = time.time()
            print(f"Starting Phase 2 encoding for {paths['preprocessed']}")
            phase2.encode_to_opus(paths['preprocessed'], paths['encoded'], bitrate='32k')
            print(f"Phase 2 completed in {time.time() - t1:.2f}s")


            # Phase 3: Postprocessing
            t1 = time.time()
            print(f"Starting Phase 3 postprocessing for {paths['encoded']}")
            phase3.postprocess_audio(paths['encoded'], paths['postprocessed'])
            print(f"Phase 3 completed in {time.time() - t1:.2f}s")

            # Evaluation Phase
            t1 = time.time()
            print("Starting evaluation phase")
            webrtc_rtt = float(request.form.get('rtt', 0.0))
            results = phase4.evaluate_audio(
                raw_path, 
                paths['postprocessed'], 
                paths['results'],
                [start_time, time.time()],
                webrtc_rtt
            )
            print(f"Evaluation completed successfully {time.time() - t1:.2f}s")

            # Final verification
            if not os.path.exists(paths['postprocessed']):
                raise PhaseProcessingError('Final processed file was not created')

            return jsonify({
                'status': 'Processing complete',
                'session_id': session_id,
                'raw_audio': f'/{raw_path}',
                'processed_audio': f'/{paths["postprocessed"]}',
                'results': results
            })

        except Exception as phase_error:
            error_msg = str(phase_error)
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            raise PhaseProcessingError(error_msg)

    except PhaseProcessingError as e:
        # Clean up all potentially created files
        cleanup_files(raw_path, *paths.values())
        return jsonify({
            'error': str(e),
            'session_id': session_id,
            'debug_info': {
                'raw_path': raw_path,
                'file_size': os.path.getsize(raw_path) if os.path.exists(raw_path) else 0,
                'files_exist': {k: os.path.exists(v) for k, v in paths.items()}
            }
        }), 500
        
    except Exception as e:
        cleanup_files(raw_path, *paths.values())
        return jsonify({
            'error': 'Unexpected processing error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/submit_mos', methods=['POST'])
def submit_mos():
    """Handle MOS survey submission with validation."""
    data = request.get_json()
    try:
        # Validate ratings
        ratings = {
            'clarity': int(data.get('clarity', 0)),
            'noise_reduction': int(data.get('noise_reduction', 0)),
            'level_matching': int(data.get('level_matching', 0))
        }
        
        if any(not (1 <= v <= 5) for v in ratings.values()):
            return jsonify({'error': 'Ratings must be between 1 and 5'}), 400
        
        # Calculate and store
        mos_average = sum(ratings.values()) / 3.0
        from dbsetup import insert_mos_survey
        insert_mos_survey(
            clarity=ratings['clarity'],
            noise_reduction=ratings['noise_reduction'],
            level_matching=ratings['level_matching'],
            audio_file=data.get('audio_file', ''),
            mos_average=mos_average
        )
        
        # return jsonify({'status': 'MOS survey saved'})
        return jsonify({'status': 'MOS survey saved', 'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Comprehensive cleanup of all generated files."""
    try:
        # Clean phase files
        for phase in ['preprocessed', 'encoded', 'postprocessed', 'evaluation']:
            for f in os.listdir(PROCESSED_FOLDER):
                if f.startswith(phase):
                    os.remove(os.path.join(PROCESSED_FOLDER, f))
        
        # Clean uploads
        cleanup_old_audio()
        
        return jsonify({
            'status': 'Cleanup completed',
            'cleaned_files': len(os.listdir(PROCESSED_FOLDER)) + len(os.listdir(UPLOAD_FOLDER))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# File serving endpoints
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    """Serve files from the uploads directory."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<path:filename>')
def serve_processed_file(filename):
    """Serve files from the processed directory."""
    return send_from_directory(PROCESSED_FOLDER, filename)

# @socketio.on('audio_stream')
# def handle_audio_stream(data):
#     """Enhanced WebRTC audio stream handler."""
#     session_id = generate_session_id()
#     raw_filename = f'webrtc_raw_{session_id}.wav'
#     raw_path = os.path.join(UPLOAD_FOLDER, raw_filename)
    
#     try:
#         # Process incoming stream
#         audio = AudioSegment(
#             data['audio'].tobytes(),
#             frame_rate=SAMPLE_RATE,
#             sample_width=2,
#             channels=1
#         )
        
#         # Ensure correct sample rate
#         if audio.frame_rate != SAMPLE_RATE:
#             audio = audio.set_frame_rate(SAMPLE_RATE)
        
#         audio.export(raw_path, format='wav')
        
#         # Process through pipeline
#         start_time = time.time()
#         paths = {
#             'preprocessed': os.path.join(PROCESSED_FOLDER, f'webrtc_preprocessed_{session_id}.wav'),
#             'encoded': os.path.join(PROCESSED_FOLDER, f'webrtc_encoded_{session_id}.opus'),
#             'postprocessed': os.path.join(PROCESSED_FOLDER, f'webrtc_postprocessed_{session_id}.wav'),
#             'results': os.path.join(PROCESSED_FOLDER, f'webrtc_evaluation_{session_id}.json')
#         }

#         phase1.preprocess_audio(raw_path, paths['preprocessed'])
#         phase2.encode_to_opus(paths['preprocessed'], paths['encoded'])
#         phase3.postprocess_audio(paths['encoded'], paths['postprocessed'])
#         end_time = time.time()
        
#         # Evaluate with WebRTC RTT
#         webrtc_rtt = float(data.get('rtt', 0.0))
#         results = phase4.evaluate_audio(
#             raw_path, 
#             paths['postprocessed'], 
#             paths['results'],
#             [start_time, end_time],
#             webrtc_rtt
#         )
        
#         # Notify client
#         socketio.emit('processing_complete', {
#             'session_id': session_id,
#             'raw_audio': f'/{raw_path}',
#             'processed_audio': f'/{paths["postprocessed"]}',
#             'results': results
#         })
#     except Exception as e:
#         socketio.emit('processing_error', {
#             'error': str(e),
#             'session_id': session_id
#         })

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """Handle WebRTC audio stream with format conversion"""
    session_id = generate_session_id()
    raw_filename = f'webrtc_raw_{session_id}.wav'
    raw_path = os.path.join(UPLOAD_FOLDER, raw_filename)
    
    try:
        # Convert received audio data to WAV
        if not convert_to_wav(data['audio'], raw_path):
            raise ValueError("Failed to convert audio to WAV format")
        
        if not is_valid_wav(raw_path):
            raise ValueError("Invalid WAV file after conversion")

        # Process through pipeline
        start_time = time.time()
        paths = {
            'preprocessed': os.path.join(PROCESSED_FOLDER, f'webrtc_preprocessed_{session_id}.wav'),
            'encoded': os.path.join(PROCESSED_FOLDER, f'webrtc_encoded_{session_id}.opus'),
            'postprocessed': os.path.join(PROCESSED_FOLDER, f'webrtc_postprocessed_{session_id}.wav'),
            'results': os.path.join(PROCESSED_FOLDER, f'webrtc_evaluation_{session_id}.json')
        }

        # Process phases
        phase1.preprocess_audio(raw_path, paths['preprocessed'])
        phase2.encode_to_opus(paths['preprocessed'], paths['encoded'])
        phase3.postprocess_audio(paths['encoded'], paths['postprocessed'])
        
        # Evaluate
        webrtc_rtt = float(data.get('rtt', 0.0))
        results = phase4.evaluate_audio(
            raw_path, 
            paths['postprocessed'], 
            paths['results'],
            [start_time, time.time()],
            webrtc_rtt
        )

        socketio.emit('processing_complete', {
            'session_id': session_id,
            'processed_audio': f'/{paths["postprocessed"]}',
            'results': results
        })

    except Exception as e:
        print(f"Processing error: {traceback.format_exc()}")
        socketio.emit('processing_error', {
            'error': f"Processing failed: {str(e)}",
            'session_id': session_id
        })
        # Clean up any created files
        for path in [raw_path] + list(paths.values()):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)