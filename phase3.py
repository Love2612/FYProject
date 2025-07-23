
# import os
# import numpy as np
# import warnings
# from scipy import signal
# from scipy.signal import butter, lfilter, sosfilt
# from pydub import AudioSegment
# import webrtcvad

# # Constants for audio processing
# TARGET_SAMPLE_RATE = 16000
# FRAME_SIZE_MS = 30
# NOISE_FLOOR_DB = -70
# MAX_SAMPLE_VALUE = 32767

# def decode_opus(input_path, apply_decode=True):
#     """Optimized Opus decoder with error handling"""
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input file not found: {input_path}")
    
#     AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"
    
#     try:
#         if not apply_decode:
#             print("[DECODE] Bypassed")
#             return np.zeros(TARGET_SAMPLE_RATE, dtype=np.int16), TARGET_SAMPLE_RATE

#         audio = AudioSegment.from_file(input_path, format="ogg", codec="libopus")
#         audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
#         samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
#         print(f"[DECODE] Success: {len(samples)/TARGET_SAMPLE_RATE:.2f}s audio")
#         return samples, audio.frame_rate
#     except Exception as e:
#         print(f"[DECODE ERROR] {str(e)[:100]}...")
#         raise

# def print_volume_stats(audio, name):
#     """Enhanced volume monitoring with LUFS-like measurement"""
#     audio_float = audio.astype(np.float32) / MAX_SAMPLE_VALUE
#     rms = np.sqrt(np.mean(audio_float**2))
#     peak = np.max(np.abs(audio_float))
#     crest = peak / (rms + 1e-10)
#     print(f"{name:<15} | RMS: {20*np.log10(rms + 1e-10):6.1f}dB | "
#           f"Peak: {20*np.log10(peak + 1e-10):6.1f}dB | "
#           f"Crest: {crest:4.1f}:1")

# def vad_denoise(audio_data, sample_rate):
#     """Advanced VAD with adaptive threshold and smoother transitions"""
#     try:
#         vad = webrtcvad.Vad()
#         vad.set_mode(3)  # Aggressive mode
        
#         if sample_rate != TARGET_SAMPLE_RATE:
#             audio_data = signal.resample_poly(
#                 audio_data, 
#                 TARGET_SAMPLE_RATE, 
#                 sample_rate
#             ).astype(np.int16)
#             sample_rate = TARGET_SAMPLE_RATE

#         frame_size = int(sample_rate * FRAME_SIZE_MS / 1000)
#         frames = len(audio_data) // frame_size
#         audio_data = audio_data[:frames * frame_size]  # Trim to whole frames
        
#         # Calculate energy per frame for adaptive threshold
#         energies = []
#         for i in range(frames):
#             frame = audio_data[i*frame_size:(i+1)*frame_size]
#             energies.append(np.mean(frame.astype(np.float32)**2))
        
#         median_energy = np.median(energies)
#         threshold = median_energy * 0.1  # Adaptive threshold
        
#         processed = np.zeros_like(audio_data)
#         for i in range(frames):
#             frame = audio_data[i*frame_size:(i+1)*frame_size]
#             frame_energy = energies[i]
            
#             if frame_energy > threshold:
#                 is_speech = vad.is_speech(frame.tobytes(), sample_rate)
#             else:
#                 is_speech = False
                
#             if is_speech:
#                 processed[i*frame_size:(i+1)*frame_size] = frame
#             else:
#                 # Apply 5ms fade out to prevent clicks
#                 fade_len = min(int(0.005 * sample_rate), frame_size)
#                 fade_out = np.linspace(1, 0, fade_len)
#                 processed[i*frame_size:i*frame_size+fade_len] = (
#                     frame[:fade_len] * fade_out
#                 )
        
#         return processed
#     except Exception as e:
#         warnings.warn(f"VAD fallback: {str(e)[:50]}")
#         return audio_data

# def dynamic_range_compression(audio_data, sample_rate, 
#                             threshold_db=-24.0, ratio=2.5, 
#                             knee_width=5.0, makeup_gain=6.0):
#     """Professional-grade DRC with lookahead and adaptive release"""
#     audio_float = audio_data.astype(np.float32) / MAX_SAMPLE_VALUE
    
#     # Envelope detector with adaptive release
#     attack_time = 0.01  # 10ms attack
#     release_time = 0.15  # 150ms release
    
#     alpha_attack = 1 - np.exp(-1/(attack_time * sample_rate))
#     alpha_release = 1 - np.exp(-1/(release_time * sample_rate))
    
#     envelope = np.zeros_like(audio_float)
#     envelope[0] = audio_float[0]**2
#     for i in range(1, len(audio_float)):
#         if audio_float[i]**2 > envelope[i-1]:
#             envelope[i] = alpha_attack * audio_float[i]**2 + (1 - alpha_attack) * envelope[i-1]
#         else:
#             # Adaptive release based on signal dynamics
#             current_release = min(release_time, 1/(i+1)) * sample_rate
#             envelope[i] = alpha_release * audio_float[i]**2 + (1 - alpha_release) * envelope[i-1]
    
#     envelope_db = 10 * np.log10(envelope + 1e-10)
    
#     # Soft knee compression curve
#     knee_low = threshold_db - knee_width/2
#     knee_high = threshold_db + knee_width/2
    
#     gain_db = np.zeros_like(envelope_db)
#     for i in range(len(envelope_db)):
#         if envelope_db[i] < knee_low:
#             gain_db[i] = makeup_gain
#         elif envelope_db[i] < knee_high:
#             x = envelope_db[i] - knee_low
#             gain_db[i] = makeup_gain + ((1/ratio - 1) * x**2) / (2 * knee_width)
#         else:
#             gain_db[i] = makeup_gain + (threshold_db + (envelope_db[i] - threshold_db)/ratio) - envelope_db[i]
    
#     # Convert to linear gain with lookahead
#     gain_linear = 10 ** (gain_db / 20)
    
#     # 5ms lookahead for smoother transitions
#     lookahead = int(0.005 * sample_rate)
#     if lookahead > 0:
#         gain_linear = np.roll(gain_linear, -lookahead)
#         gain_linear[-lookahead:] = gain_linear[-lookahead-1]
    
#     # Apply compression
#     compressed = audio_float * gain_linear
    
#     # Adaptive limiter to prevent clipping
#     peak = np.max(np.abs(compressed))
#     if peak > 0.95:  # -0.5dBFS threshold
#         compressed = compressed * (0.95 / peak)
    
#     return np.clip(compressed * MAX_SAMPLE_VALUE, -MAX_SAMPLE_VALUE, MAX_SAMPLE_VALUE).astype(np.int16)

# def presence_boost(audio_data, sample_rate):
#     """Multiband presence enhancement"""
#     nyquist = sample_rate / 2
#     audio_float = audio_data.astype(np.float32) / MAX_SAMPLE_VALUE
    
#     # Stage 1: Presence boost (3-5kHz)
#     sos_presence = butter(2, [3000/nyquist, 5000/nyquist], btype='band', output='sos')
#     presence_band = sosfilt(sos_presence, audio_float) * 1.4  # +3dB boost
    
#     # Stage 2: Air boost (8-12kHz)
#     air_low = 2000
#     air_high = 4500
#     if air_high < nyquist:
#         sos_air = butter(2, [air_low/nyquist, air_high/nyquist], btype='band', output='sos')
#         air_band = sosfilt(sos_air, audio_float) * 0.3
#     else:
#         air_band = np.zeros_like(audio_float)  # fallback if too low sample rate
    
#     # Mix with dry signal
#     processed = audio_float + 0.3*presence_band + 0.1*air_band
    
#     # Output limiting
#     peak = np.max(np.abs(processed))
#     if peak > 0.95:
#         processed = processed * (0.95 / peak)
    
#     return np.clip(processed * MAX_SAMPLE_VALUE, -MAX_SAMPLE_VALUE, MAX_SAMPLE_VALUE).astype(np.int16)

# def postprocess_audio(input_path, output_path, reference_path=None, 
#                      sample_rate=TARGET_SAMPLE_RATE, **kwargs):
#     """Optimized master processing chain"""
#     # Decode audio
#     audio_data, sample_rate = decode_opus(input_path, kwargs.get('apply_decode', True))
#     print_volume_stats(audio_data, "Original")
    
#     # Process each stage
#     processing_steps = [
#         ('VAD', kwargs.get('apply_vad', True), vad_denoise),
#         ('DRC', kwargs.get('apply_drc', True), dynamic_range_compression),
#         ('EchoCancel', kwargs.get('apply_echo', True) and reference_path, 
#          lambda x, sr: echo_cancellation(x, sr, load_reference(reference_path, sr))),
#         ('PresenceBoost', kwargs.get('apply_boost', True), presence_boost)
#     ]
    
#     for name, apply, func in processing_steps:
#         if apply:
#             audio_data = func(audio_data, sample_rate)
#             print_volume_stats(audio_data, f"After {name}")
    
#     # Final loudness normalization
#     audio_float = audio_data.astype(np.float32) / MAX_SAMPLE_VALUE
#     integrated_loudness = 10 * np.log10(np.mean(audio_float**2) + 1e-10)
#     target_lufs = -16.0  # Broadcast standard
    
#     if integrated_loudness < target_lufs:
#         gain = 10 ** ((target_lufs - integrated_loudness) / 20)
#         audio_float = audio_float * gain
#         print(f"[LOUDNESS] Applied {20*np.log10(gain):.1f}dB gain")
    
#     # Convert and export
#     final_audio = np.clip(audio_float * MAX_SAMPLE_VALUE, -MAX_SAMPLE_VALUE, MAX_SAMPLE_VALUE)
#     export_audio(final_audio.astype(np.int16), sample_rate, output_path)
    
#     return output_path

# def load_reference(path, sample_rate):
#     """Load reference audio for echo cancellation"""
#     ref = AudioSegment.from_wav(path).set_channels(1).set_frame_rate(sample_rate)
#     return np.array(ref.get_array_of_samples(), dtype=np.int16)

# def export_audio(audio_data, sample_rate, output_path):
#     """Optimized audio export"""
#     audio = AudioSegment(
#         audio_data.tobytes(),
#         frame_rate=sample_rate,
#         sample_width=2,
#         channels=1
#     )
#     audio.export(output_path, format="wav", parameters=["-ar", str(sample_rate)])
#     print(f"[EXPORT] Saved {output_path}")

# if __name__ == "__main__":
#     input_audio = "static/processed/encoded_audio.wav"
#     output_audio = "static/processed/postprocessed_audio.wav"
    
#     try:
#         postprocess_audio(
#             input_audio,
#             output_audio,
#             reference_path=None,
#             apply_vad=True,
#             apply_drc=False,
#             apply_echo=False,
#             apply_boost=False
#         )
#     except Exception as e:
#         print(f"[FATAL] {str(e)}")
#         raise




import os
import numpy as np
import warnings
from scipy import signal
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import webrtcvad

# Constants
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLE_VALUE = 32767
FRAME_SIZE_MS = 30

AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"

### ========== DECODING ========== ###

def decode_opus(input_path):
    """Decode Opus to PCM for processing."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        audio = AudioSegment.from_file(input_path, format="ogg", codec="libopus")
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        print(f"[DECODE] {input_path} → {len(samples)/TARGET_SAMPLE_RATE:.2f}s audio decoded")
        return samples, TARGET_SAMPLE_RATE
    except Exception as e:
        raise RuntimeError(f"[DECODE ERROR] {e}")

### ========== POSTPROCESSING TECHNIQUES ========== ###

def vad_denoise(audio_data, sample_rate):
    """Voice Activity Detection (VAD) for denoising."""
    vad = webrtcvad.Vad(3)  # Aggressive mode
    frame_len = int(sample_rate * FRAME_SIZE_MS / 1000)
    num_frames = len(audio_data) // frame_len
    audio_data = audio_data[:num_frames * frame_len]

    processed = np.zeros_like(audio_data)
    for i in range(num_frames):
        frame = audio_data[i*frame_len:(i+1)*frame_len]
        if vad.is_speech(frame.tobytes(), sample_rate):
            processed[i*frame_len:(i+1)*frame_len] = frame
    return processed

def dynamic_range_compression(audio_data, sample_rate, threshold_db=-24.0, ratio=2.5):
    """Apply soft-knee compression to even out volume."""
    audio_float = audio_data.astype(np.float32) / MAX_SAMPLE_VALUE
    rms = np.sqrt(np.mean(audio_float**2))
    current_db = 20 * np.log10(rms + 1e-10)

    if current_db < threshold_db:
        return audio_data  # no compression needed

    gain = 10 ** ((threshold_db - current_db) / (20 * ratio))
    compressed = audio_float * gain
    peak = np.max(np.abs(compressed))
    if peak > 1.0:
        compressed *= (0.95 / peak)

    return np.clip(compressed * MAX_SAMPLE_VALUE, -MAX_SAMPLE_VALUE, MAX_SAMPLE_VALUE).astype(np.int16)

def echo_cancellation(audio_data, sample_rate, reference=None):
    """Dummy placeholder: Echo cancellation would require adaptive filters or reference signal."""
    print("[ECHO] Placeholder applied (no-op)")
    return audio_data

### ========== EXPORT ========== ###

def export_audio(audio_data, sample_rate, output_path):
    """Save processed PCM audio as WAV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    audio.export(output_path, format="wav")
    print(f"[EXPORT] Processed audio saved to {output_path}")

### ========== MAIN CHAIN ========== ###

def postprocess_audio(input_path, output_path,
                      apply_denoise=True,
                      apply_drc=True,
                      apply_echo=False):
    """Run postprocessing chain after decoding Opus."""
    audio_data, sample_rate = decode_opus(input_path)

    if apply_denoise:
        print("[POST] Applying VAD-based denoising...")
        audio_data = vad_denoise(audio_data, sample_rate)

    if apply_drc:
        print("[POST] Applying dynamic range compression...")
        audio_data = dynamic_range_compression(audio_data, sample_rate)

    if apply_echo:
        print("[POST] Applying echo cancellation...")
        audio_data = echo_cancellation(audio_data, sample_rate)

    export_audio(audio_data, sample_rate, output_path)
    return output_path

### ========== TEST HARNESS ========== ###

if __name__ == "__main__":
    input_file = "static/processed/encoded_audio.ogg"
    output_file = "static/processed/postprocessed_audio.wav"

    try:
        postprocess_audio(
            input_path=input_file,
            output_path=output_file,
            apply_denoise=False,
            apply_drc=False,
            apply_echo=False # Here False means skip while True means apply
        )
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
