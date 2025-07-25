

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
    print("[ECHO] cancellation")
    return audio_data


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
