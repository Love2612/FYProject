

# # import numpy as np
# # from scipy import signal
# # import soundfile as sf
# # import librosa
# # import noisereduce as nr
# # import os

# # def soft_clip(audio_data, threshold=0.9):
# #     """Apply soft clipping to prevent distortion."""
# #     audio_float = audio_data.astype(np.float32)
# #     audio_float = np.tanh(audio_float / threshold) * threshold
# #     return audio_float

# # def pre_filter(audio_data, sample_rate):
# #     """Apply bandpass filter with dynamic cutoff frequencies to remove extreme frequencies."""
# #     nyquist = sample_rate / 2
# #     low_freq = max(100, 0.1 * nyquist)
# #     high_freq = min(8000, 0.9 * nyquist)

# #     low = low_freq / nyquist
# #     high = high_freq / nyquist

# #     if low <= 0 or high >= 1:
# #         print(f"[WARNING] Invalid filter frequencies: low={low_freq}Hz, high={high_freq}Hz. Skipping pre-filter.")
# #         return audio_data

# #     try:
# #         sos = signal.butter(4, [low, high], btype='band', output='sos')
# #         return signal.sosfilt(sos, audio_data)
# #     except Exception as e:
# #         print(f"[ERROR] Pre-filter failed: {e}")
# #         return audio_data

# # def noise_suppression(audio_data, sample_rate):
# #     """Denoise using noisereduce with adjusted parameters."""
# #     try:
# #         noise_clip = audio_data[:int(sample_rate * 0.25)]  # Shorter noise profile
# #         reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, y_noise=noise_clip, prop_decrease=0.7)
# #         return soft_clip(reduced_noise)
# #     except Exception as e:
# #         print(f"[ERROR] Noise suppression failed: {e}")
# #         return audio_data

# # def noise_gate(audio_data, threshold=0.01):
# #     """Apply noise gate to suppress low-level noise."""
# #     audio_float = audio_data.astype(np.float32)
# #     audio_float[np.abs(audio_float) < threshold] = 0
# #     return audio_float

# # def gain_control(audio_data, target_rms_db=-20.0, limit_peak=True):
# #     """
# #     Apply gentle gain normalization with noise gate and peak limiting.
# #     """
# #     # Step 1: Apply noise gate
# #     audio_data = noise_gate(audio_data)

# #     # Step 2: Compute RMS
# #     rms = np.sqrt(np.mean(audio_data ** 2))
# #     if rms < 1e-6:
# #         print("[GAIN] Signal too quiet. Skipping gain.")
# #         return audio_data

# #     # Step 3: Calculate gain in dB
# #     current_rms_db = 20 * np.log10(rms)
# #     gain_db = target_rms_db - current_rms_db
# #     print(f"[GAIN] Current RMS dBFS: {current_rms_db:.2f}, Target: {target_rms_db}, Applying: {gain_db:.2f} dB")

# #     # Step 4: Convert to linear and apply
# #     gain_linear = 10 ** (gain_db / 20)
# #     boosted = audio_data * gain_linear

# #     # Step 5: Limit peak to avoid distortion
# #     if limit_peak:
# #         peak = np.max(np.abs(boosted))
# #         if peak > 1.0:
# #             boosted = boosted / peak
# #             print(f"[LIMIT] Peak after gain was {peak:.2f}. Normalised to 1.0.")

# #     return np.clip(boosted, -1.0, 1.0)  # Final hard clip for safety


# # def equalization(audio_data, sample_rate):
# #     """Apply gentle high-shelf EQ and low-pass filter with adaptive cutoff frequencies."""
# #     nyquist = sample_rate / 2
# #     air_low_freq = max(600, 0.3 * nyquist)
# #     air_high_freq = min(12000, 0.8 * nyquist)

# #     if air_high_freq >= nyquist or air_low_freq >= nyquist:
# #         print(f"[WARNING] EQ band invalid (>{nyquist} Hz). Skipping EQ.")
# #         return audio_data

# #     try:
# #         # Band filter to gently boost air band
# #         sos = signal.butter(1, [air_low_freq / nyquist, air_high_freq / nyquist], btype='band', output='sos')
# #         air_boost = signal.sosfilt(sos, audio_data) * 1.1
# #         enhanced = audio_data + air_boost

# #         # Low-pass filter to remove harsh edges
# #         cutoff = min(0.99 * nyquist, 7900)  # Ensure cutoff < Nyquist
# #         sos_low = signal.butter(4, cutoff / nyquist, btype='low', output='sos')
# #         enhanced = signal.sosfilt(sos_low, enhanced)

# #         return np.clip(soft_clip(enhanced), -1.0, 1.0)
# #     except Exception as e:
# #         print(f"[ERROR] Equalization failed: {e}")
# #         return audio_data

# # def preprocess_audio(input_path, output_path, sample_rate=16000, apply_ns=True, apply_gain=True, apply_eq=True):
# #     """Main function to preprocess audio with minimal distortion."""
# #     if not os.path.exists(input_path):
# #         raise FileNotFoundError(f"Input audio file {input_path} not found")

# #     try:
# #         audio_data, original_sample_rate = sf.read(input_path)
# #     except Exception as e:
# #         raise ValueError(f"Failed to load audio file {input_path}: {e}")

# #     print(f"[DEBUG] Original sample rate: {original_sample_rate}, Shape: {audio_data.shape}")
# #     print(f"[DEBUG] Max amplitude before processing: {np.max(np.abs(audio_data))}")

# #     if audio_data.size == 0:
# #         raise ValueError("Loaded audio data is empty")

# #     # Convert stereo to mono
# #     try:
# #         if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
# #             audio_data = np.mean(audio_data, axis=1)
# #         elif len(audio_data.shape) > 1 and audio_data.shape[1] == 1:
# #             audio_data = audio_data[:, 0]
# #     except IndexError as e:
# #         print(f"[ERROR] Shape handling failed: {e}. Assuming mono audio.")

# #     print(f"[DEBUG] After mono conversion - Shape: {audio_data.shape}, Max amplitude: {np.max(np.abs(audio_data))}")

# #     current_sample_rate = original_sample_rate
# #     if current_sample_rate != sample_rate:
# #         try:
# #             print(f"[DEBUG] Resampling from {current_sample_rate} to {sample_rate} Hz")
# #             audio_data = librosa.resample(audio_data, orig_sr=current_sample_rate, target_sr=sample_rate)
# #             current_sample_rate = sample_rate
# #         except Exception as e:
# #             print(f"[ERROR] Resampling failed: {e}")
# #             return input_path

# #     print(f"[DEBUG] After resampling - Shape: {audio_data.shape}, Max amplitude: {np.max(np.abs(audio_data))}")

# #     # Save checkpoint (optional)
# #     # sf.write(output_path.replace('.wav', '_resampled.wav'), audio_data.astype(np.float32), current_sample_rate)

# #     # Pre-filter
# #     audio_data = pre_filter(audio_data, current_sample_rate)

# #     if apply_ns:
# #         audio_data = noise_suppression(audio_data, current_sample_rate)

# #     if apply_gain:
# #         audio_data = gain_control(audio_data, target_rms_db=-20.0)

# #     if apply_eq:
# #         audio_data = equalization(audio_data, current_sample_rate)

# #     # Final clipping and normalisation
# #     peak = np.max(np.abs(audio_data))
# #     if peak > 1.0:
# #         audio_data /= peak  # Normalize to avoid clipping

# #     # Final output: save as float32 for safety
# #     sf.write(output_path, audio_data.astype(np.float32), current_sample_rate)
# #     print(f"[OUTPUT] Preprocessed audio saved to {output_path}")

# #     return output_path

# # if __name__ == "__main__":
# #     input_audio = "uploads/test_audio.ogg"
# #     output_audio = "static/processed/preprocessed_audio.wav"
# #     try:
# #         preprocess_audio(input_audio, output_audio, apply_ns=False, apply_gain=True, apply_eq=False)

# #     except Exception as e:
# #         print(f"Error during preprocessing: {e}")



# import numpy as np
# from scipy import signal
# import soundfile as sf
# import librosa
# import noisereduce as nr
# import os
# import pyloudnorm as pyln

# def soft_clip(audio_data, threshold=0.9):
#     """Apply soft clipping to prevent distortion."""
#     audio_float = audio_data.astype(np.float32)
#     audio_float = np.tanh(audio_float / threshold) * threshold
#     return audio_float

# def pre_filter(audio_data, sample_rate):
#     """Apply bandpass filter with dynamic cutoff frequencies to remove extreme frequencies."""
#     nyquist = sample_rate / 2
#     low_freq = max(100, 0.1 * nyquist)
#     high_freq = min(8000, 0.9 * nyquist)

#     low = low_freq / nyquist
#     high = high_freq / nyquist

#     if low <= 0 or high >= 1:
#         print(f"[WARNING] Invalid filter frequencies: low={low_freq}Hz, high={high_freq}Hz. Skipping pre-filter.")
#         return audio_data

#     try:
#         sos = signal.butter(4, [low, high], btype='band', output='sos')
#         return signal.sosfilt(sos, audio_data)
#     except Exception as e:
#         print(f"[ERROR] Pre-filter failed: {e}")
#         return audio_data

# def noise_suppression(audio_data, sample_rate):
#     """Denoise using noisereduce with adjusted parameters."""
#     try:
#         noise_clip = audio_data[:int(sample_rate * 0.25)]  # Shorter noise profile
#         reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, y_noise=noise_clip, prop_decrease=0.7)
#         return soft_clip(reduced_noise)
#     except Exception as e:
#         print(f"[ERROR] Noise suppression failed: {e}")
#         return audio_data

# def noise_gate(audio_data, threshold=0.01):
#     """Apply noise gate to suppress low-level noise."""
#     audio_float = audio_data.astype(np.float32)
#     audio_float[np.abs(audio_float) < threshold] = 0
#     return audio_float

# # def gain_control(audio_data, target_rms_db=-20.0, max_gain_db=6):
# #     """
# #     Safely normalise audio to a target RMS level, with strict peak limiting and no clipping.
# #     """
# #     # Remove NaNs or infs
# #     audio_data = np.nan_to_num(audio_data)

# #     # Step 1: Apply noise gate
# #     gated = noise_gate(audio_data)

# #     # Step 2: Compute RMS
# #     rms = np.sqrt(np.mean(gated ** 2))
# #     if rms < 1e-6:
# #         print("[GAIN] Too quiet, skipping gain.")
# #         return gated

# #     # Step 3: Calculate gain
# #     current_rms_db = 20 * np.log10(rms)
# #     gain_db = target_rms_db - current_rms_db
# #     gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)  # Limit gain change
# #     print(f"[GAIN] RMS: {current_rms_db:.2f} dBFS → {target_rms_db} dBFS | Applying: {gain_db:.2f} dB")

# #     # Step 4: Apply gain
# #     gain_linear = 10 ** (gain_db / 20)
# #     boosted = gated * gain_linear

# #     # Step 5: True peak limit to avoid clipping
# #     peak = np.max(np.abs(boosted))
# #     if peak > 0.99:
# #         boosted = boosted / peak * 0.99
# #         print(f"[LIMIT] Peak {peak:.2f} scaled to 0.99")

# #     # Step 6: Clip any strays and return
# #     return np.clip(boosted, -1.0, 1.0)

# # def smart_gain(audio_data, target_rms_db=-20.0, max_gain_db=6.0):
# #     rms = np.sqrt(np.mean(audio_data**2))
# #     peak = np.max(np.abs(audio_data))

# #     current_rms_db = 20 * np.log10(rms + 1e-10)
# #     gain_db = target_rms_db - current_rms_db

# #     # Constrain based on peak margin
# #     headroom_db = 20 * np.log10(1.0 / (peak + 1e-6))
# #     gain_db = min(gain_db, headroom_db - 0.5)  # Leave 0.5 dB headroom
# #     gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)

# #     print(f"[SMART GAIN] RMS: {current_rms_db:.2f} dB, Peak: {peak:.2f}, Gain: {gain_db:.2f} dB")

# #     gain = 10 ** (gain_db / 20)
# #     boosted = audio_data * gain

# #     return np.clip(boosted, -1.0, 1.0)

# def gain_control_lufs(audio_data, sample_rate, target_lufs=-23.0):
#     meter = pyln.Meter(sample_rate)
#     loudness = meter.integrated_loudness(audio_data)

#     print(f"[LUFS] Current: {loudness:.2f} LUFS → Target: {target_lufs:.2f} LUFS")

#     normalized_audio = pyln.normalize.loudness(audio_data, loudness, target_lufs)

#     # Limit peaks to avoid distortion
#     peak = np.max(np.abs(normalized_audio))
#     if peak > 0.99:
#         normalized_audio = normalized_audio / peak * 0.99
#         print(f"[LUFS LIMIT] Peak scaled from {peak:.2f} to 0.99")

#     return np.clip(normalized_audio, -1.0, 1.0)

# def apply_limiter(audio_data, ceiling=0.95):
#     """True peak limiter using normalization + soft clip"""
#     peak = np.max(np.abs(audio_data))
#     if peak > ceiling:
#         audio_data = (audio_data / peak) * ceiling  # Normalize to ceiling
#         print(f"[LIMITER] Audio scaled down from peak {peak:.3f} to ceiling {ceiling}")
#     return np.clip(audio_data, -1.0, 1.0)


# def equalization(audio_data, sample_rate):
#     """Apply gentle high-shelf EQ and low-pass filter with adaptive cutoff frequencies."""
#     nyquist = sample_rate / 2
#     air_low_freq = max(600, 0.3 * nyquist)
#     air_high_freq = min(12000, 0.8 * nyquist)

#     if air_high_freq >= nyquist or air_low_freq >= nyquist:
#         print(f"[WARNING] EQ band invalid (>{nyquist} Hz). Skipping EQ.")
#         return audio_data

#     try:
#         sos = signal.butter(1, [air_low_freq / nyquist, air_high_freq / nyquist], btype='band', output='sos')
#         air_boost = signal.sosfilt(sos, audio_data) * 1.1
#         enhanced = audio_data + air_boost

#         cutoff = min(0.99 * nyquist, 7900)
#         sos_low = signal.butter(4, cutoff / nyquist, btype='low', output='sos')
#         enhanced = signal.sosfilt(sos_low, enhanced)

#         return np.clip(soft_clip(enhanced), -1.0, 1.0)
#     except Exception as e:
#         print(f"[ERROR] Equalization failed: {e}")
#         return audio_data

# def preprocess_audio(input_path, output_path, sample_rate=16000, apply_ns=True, apply_gain=True, apply_eq=True):
#     """Main function to preprocess audio with minimal distortion."""
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input audio file {input_path} not found")

#     try:
#         audio_data, original_sample_rate = sf.read(input_path)
#     except Exception as e:
#         raise ValueError(f"Failed to load audio file {input_path}: {e}")

#     print(f"[DEBUG] Original sample rate: {original_sample_rate}, Shape: {audio_data.shape}")
#     print(f"[DEBUG] Max amplitude before processing: {np.max(np.abs(audio_data))}")

#     if audio_data.size == 0:
#         raise ValueError("Loaded audio data is empty")

#     try:
#         if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
#             audio_data = np.mean(audio_data, axis=1)
#         elif len(audio_data.shape) > 1 and audio_data.shape[1] == 1:
#             audio_data = audio_data[:, 0]
#     except IndexError as e:
#         print(f"[ERROR] Shape handling failed: {e}. Assuming mono audio.")

#     print(f"[DEBUG] After mono conversion - Shape: {audio_data.shape}, Max amplitude: {np.max(np.abs(audio_data))}")

#     current_sample_rate = original_sample_rate
#     if current_sample_rate != sample_rate:
#         try:
#             print(f"[DEBUG] Resampling from {current_sample_rate} to {sample_rate} Hz")
#             audio_data = librosa.resample(audio_data, orig_sr=current_sample_rate, target_sr=sample_rate)
#             current_sample_rate = sample_rate
#         except Exception as e:
#             print(f"[ERROR] Resampling failed: {e}")
#             return input_path

#     print(f"[DEBUG] After resampling - Shape: {audio_data.shape}, Max amplitude: {np.max(np.abs(audio_data))}")

#     audio_data = pre_filter(audio_data, current_sample_rate)

#     if apply_ns:
#         audio_data = noise_suppression(audio_data, current_sample_rate)

#     if apply_gain:
#         # audio_data = gain_control(audio_data, target_rms_db=-20.0)
#         # audio_data = smart_gain(audio_data)
#         audio_data = gain_control_lufs(audio_data, current_sample_rate, target_lufs=-23.0)

        
#     # audio_data = apply_limiter(audio_data)

#     if apply_eq:
#         audio_data = equalization(audio_data, current_sample_rate)

#     # peak = np.max(np.abs(audio_data))
#     # if peak > 1.0:
#     #     audio_data /= peak

#     # audio_data *= 0.95  # <-- Safety margin to avoid codec distortion
    
    
    
#     sf.write(output_path, audio_data.astype(np.float32), current_sample_rate)
#     print(f"[OUTPUT] Preprocessed audio saved to {output_path}")

#     return output_path

# if __name__ == "__main__":
#     input_audio = "uploads/test_audio.ogg"
#     output_audio = "static/processed/preprocessed_audio.wav"
#     try:
#         preprocess_audio(input_audio, output_audio, apply_ns=True, apply_gain=True, apply_eq=True)
#     except Exception as e:
#         print(f"Error during preprocessing: {e}")









import numpy as np
from scipy import signal
import soundfile as sf
import librosa
import noisereduce as nr
import pyloudnorm as pyln
import os

### ========== FILTERS ========== ###

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


### ========== MAIN ENGINE ========== ###

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


### ========== TEST HARNESS ========== ###

if __name__ == "__main__":
    # Manual testing: enable/disable any filter
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
