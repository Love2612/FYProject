import numpy as np
from scipy import signal
from scipy.io import wavfile
import pydub
import os
from scipy.signal import stft, istft, butter, lfilter


def noise_suppression(audio_data, sample_rate):
    """Denoise using high-pass + soft spectral gate (non-AI)."""
    # Step 1: High-pass filter to remove low rumble (< 100 Hz)
    def highpass_filter(data, cutoff=100, fs=16000):
        nyq = fs / 2
        b, a = butter(2, cutoff / nyq, btype='high')
        return lfilter(b, a, data)

    audio_filtered = highpass_filter(audio_data, cutoff=100, fs=sample_rate)

    # Step 2: Soft spectral gate
    f, t, Zxx = stft(audio_filtered, fs=sample_rate, nperseg=512)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise as the quietest 10% of frames
    noise_est = np.percentile(magnitude, 10, axis=1, keepdims=True)

    # Soft gate: suppress only when signal is close to noise
    suppression_mask = magnitude < (1.5 * noise_est)
    cleaned_magnitude = np.where(suppression_mask, magnitude * 0.3, magnitude)

    # Reconstruct signal
    Zxx_clean = cleaned_magnitude * np.exp(1j * phase)
    _, audio_denoised = istft(Zxx_clean, fs=sample_rate)

    # Normalize
    peak = np.max(np.abs(audio_denoised))
    if peak > 0:
        audio_denoised = audio_denoised / peak * 32767

    return audio_denoised.astype(np.int16)


def gain_control(audio_data, target_rms_db=-25.0):
    """Apply gentle gain normalization using float RMS scaling."""
    # Convert to float32 for safe processing
    audio_float = audio_data.astype(np.float32)

    # Compute RMS in dBFS
    rms = np.sqrt(np.mean(audio_float ** 2))
    if rms < 1e-6:
        print("[GAIN] Signal is silent or too quiet. Skipping gain.")
        return audio_data

    current_rms_db = 20 * np.log10(rms / 32767)
    gain_db = target_rms_db - current_rms_db

    print(f"[GAIN] Current RMS dBFS: {current_rms_db:.2f}, Target: {target_rms_db}, Applying: {gain_db:.2f} dB")

    # Convert gain from dB to linear scale
    gain_linear = 10 ** (gain_db / 20)
    boosted = audio_float * gain_linear

    # Clip and return as int16
    boosted = np.clip(boosted, -32768, 32767)
    return boosted.astype(np.int16)



def equalization(audio_data, sample_rate):
    """Apply equalization to enhance speech frequencies (300Hz - 3400Hz)."""
    # Design a bandpass filter for 300Hz - 3400Hz
    nyquist = sample_rate / 2
    low_freq = 300 / nyquist
    high_freq = 3400 / nyquist
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    
    # Apply filter
    audio_eq = signal.lfilter(b, a, audio_data)
    return audio_eq


def preprocess_audio(input_path, output_path, sample_rate=16000,
                     apply_ns=True, apply_gain=True, apply_eq=True):
    """Main function to preprocess audio with optional noise suppression, gain control, and EQ."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file {input_path} not found")
    
    original_sample_rate, audio_data = wavfile.read(input_path)
    
    print(f"[DEBUG] Original sample rate: {original_sample_rate}, Shape: {audio_data.shape}")
    print(f"[DEBUG] Max amplitude before processing: {np.max(np.abs(audio_data))}")
    if np.isnan(audio_data).any():
        print("[WARNING] NaNs found in input audio!")

    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if original_sample_rate != 16000:
        audio_data = signal.resample_poly(audio_data, 16000, original_sample_rate)

    # Step 1: Noise Suppression
    if apply_ns:
        audio_data = noise_suppression(audio_data, 16000)
        print(f"[DEBUG] After noise suppression - max: {np.max(np.abs(audio_data))}")
    else:
        print("[DEBUG] Noise suppression bypassed.")

    # Step 2: Gain Control
    if apply_gain:
        audio_data = gain_control(audio_data, target_rms_db=-20.0)
        print(f"[DEBUG] After gain control - max: {np.max(np.abs(audio_data))}")
    else:
        print("[DEBUG] Gain control bypassed.")

    # Step 3: Equalization
    if apply_eq:
        audio_data = equalization(audio_data, 16000)
        print(f"[DEBUG] After equalization - max: {np.max(np.abs(audio_data))}")
    else:
        print("[DEBUG] Equalization bypassed.")

    # Normalize output if too quiet
    peak = np.max(np.abs(audio_data))
    if peak < 1000:
        print(f"[WARNING] Audio too quiet (peak={peak}). Applying normalization.")
        audio_data = audio_data * (32767 / (peak + 1e-6))

    audio_data = np.clip(audio_data, -32768, 32767)
    wavfile.write(output_path, 16000, audio_data.astype(np.int16))
    
    return output_path


if __name__ == "__main__":
    # Example usage for testing
    input_audio = "uploads/test_audio.wav"
    output_audio = "static/processed/preprocessed_audio.wav"
    try:
        # Run with all stages enabled
        preprocess_audio(input_audio, output_audio)

        # Isolation test: Only raw input
        # preprocess_audio(input_audio, output_audio, apply_ns=False, apply_gain=False, apply_eq=False)

        # Isolation test: Only noise suppression
        # preprocess_audio(input_audio, output_audio, apply_ns=True, apply_gain=False, apply_eq=False)

        # Isolation test: Only gain control
        # preprocess_audio(input_audio, output_audio, apply_ns=False, apply_gain=True, apply_eq=False)

        # Isolation test: Only equalization
        # preprocess_audio(input_audio, output_audio, apply_ns=False, apply_gain=False, apply_eq=True)

        # Test this one first:
        # preprocess_audio(input_audio, output_audio, apply_ns=False, apply_gain=False, apply_eq=False)
        
        
        # preprocess_audio(input_audio, output_audio)
        print(f"Preprocessed audio saved to {output_audio}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")