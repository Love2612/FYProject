
import numpy as np
from scipy import signal
import soundfile as sf
import librosa
import noisereduce as nr
import pyloudnorm as pyln
import os


def soft_clip(audio_data, threshold=0.9):
    """Prevent harsh clipping by applying soft-knee compression."""
    audio_float = audio_data.astype(np.float32)
    audio_float = np.tanh(audio_float / threshold) * threshold
    return audio_float

def pre_filter(audio_data, sample_rate):
    """Apply bandpass filter to limit frequencies between ~100Hz and 8000Hz."""
    nyquist = sample_rate / 2
    low_freq = max(100, 0.1 * nyquist)
    high_freq = min(8000, 0.9 * nyquist)

    low = low_freq / nyquist
    high = high_freq / nyquist

    if low <= 0 or high >= 1:
        print(f"[WARNING] Invalid filter frequencies: low={low_freq}Hz, high={high_freq}Hz. Skipping.")
        return audio_data

    try:
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfilt(sos, audio_data)
    except Exception as e:
        print(f"[ERROR] Bandpass filter failed: {e}")
        return audio_data

def noise_suppression(audio_data, sample_rate):
    """Use noise profile subtraction to reduce background noise."""
    try:
        noise_clip = audio_data[:int(sample_rate * 0.25)]
        reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, y_noise=noise_clip, prop_decrease=0.7)
        return soft_clip(reduced_noise)
    except Exception as e:
        print(f"[ERROR] Noise suppression failed: {e}")
        return audio_data

def gain_control_lufs(audio_data, sample_rate, target_lufs=-23.0):
    """Normalize audio to a consistent loudness level using LUFS standard."""
    try:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio_data)
        print(f"[LUFS] Current: {loudness:.2f} LUFS → Target: {target_lufs:.2f} LUFS")

        normalized = pyln.normalize.loudness(audio_data, loudness, target_lufs)
        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized / peak * 0.99
            print(f"[LUFS LIMIT] Peak limited from {peak:.2f} to 0.99")
        return np.clip(normalized, -1.0, 1.0)
    except Exception as e:
        print(f"[ERROR] Gain control failed: {e}")
        return audio_data

def equalization(audio_data, sample_rate):
    """Boost air band and apply low-pass filtering for speech clarity."""
    nyquist = sample_rate / 2
    low_band = max(600, 0.3 * nyquist)
    high_band = min(12000, 0.8 * nyquist)

    if high_band >= nyquist or low_band >= nyquist:
        print(f"[WARNING] EQ band invalid (> Nyquist). Skipping EQ.")
        return audio_data

    try:
        sos = signal.butter(1, [low_band / nyquist, high_band / nyquist], btype='band', output='sos')
        air_boost = signal.sosfilt(sos, audio_data) * 1.1
        enhanced = audio_data + air_boost

        low_cutoff = min(7900, 0.99 * nyquist)
        sos_low = signal.butter(4, low_cutoff / nyquist, btype='low', output='sos')
        enhanced = signal.sosfilt(sos_low, enhanced)

        return np.clip(soft_clip(enhanced), -1.0, 1.0)
    except Exception as e:
        print(f"[ERROR] Equalization failed: {e}")
        return audio_data


def preprocess_audio(input_path, output_path,
                     sample_rate=16000,
                     bypass_ns=False,
                     bypass_gain=False,
                     bypass_eq=False):
    """Core preprocessing function with bypass toggles."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    try:
        audio_data, original_sr = sf.read(input_path)
    except Exception as e:
        raise ValueError(f"Cannot load audio: {e}")

    if audio_data.size == 0:
        raise ValueError("Empty audio file")

    # Convert stereo to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample
    if original_sr != sample_rate:
        try:
            print(f"[INFO] Resampling from {original_sr} → {sample_rate} Hz")
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=sample_rate)
        except Exception as e:
            print(f"[ERROR] Resampling failed: {e}")
            return input_path

    # Pre-filtering always applied
    audio_data = pre_filter(audio_data, sample_rate)

    if not bypass_ns:
        audio_data = noise_suppression(audio_data, sample_rate)

    if not bypass_gain:
        audio_data = gain_control_lufs(audio_data, sample_rate, target_lufs=-23.0)

    if not bypass_eq:
        audio_data = equalization(audio_data, sample_rate)

    sf.write(output_path, audio_data.astype(np.float32), sample_rate)
    print(f"[OUTPUT] Saved processed audio to: {output_path}")
    return output_path



if __name__ == "__main__":
    # Manual testing
    input_audio = "uploads/test_audio.ogg"
    output_audio = "static/processed/preprocessed_audio.wav"

    try:
        preprocess_audio(
            input_audio,
            output_audio,
            bypass_ns=True,   # Applies Noise suppression
            bypass_gain=True, # Applies Gain control
            bypass_eq=True    # Applies Equalization (If True then it has skipped all the processes)
        )
    except Exception as e:
        print(f"[FAILURE] Preprocessing failed: {e}")
