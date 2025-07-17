

# import os
# import numpy as np
# from scipy.io import wavfile
# import json
# import time
# from scipy import signal


# def calculate_snr(raw_audio, processed_audio):
#     """Calculate Signal-to-Noise Ratio (SNR) in dB."""
#     signal_power = np.mean(raw_audio ** 2)
#     noise = raw_audio - processed_audio
#     noise_power = np.mean(noise ** 2) + 1e-10  # Avoid division by zero
#     snr = 10 * np.log10(signal_power / noise_power)
#     return snr

# def calculate_peaq_approximation(raw_audio, processed_audio, sample_rate):
#     """Approximate Perceptual Evaluation of Audio Quality (PEAQ) using frequency analysis."""
#     # Compute power spectral density for raw and processed audio
#     freqs_raw, psd_raw = signal.welch(raw_audio, fs=sample_rate, nperseg=1024)
#     freqs_proc, psd_proc = signal.welch(processed_audio, fs=sample_rate, nperseg=1024)
    
#     # Focus on speech frequencies (300Hz - 3400Hz)
#     speech_mask = (freqs_raw >= 300) & (freqs_raw <= 3400)
#     raw_speech_power = np.mean(psd_raw[speech_mask])
#     proc_speech_power = np.mean(psd_proc[speech_mask])
    
#     # Simplified PEAQ: Ratio of speech power retention (higher is better)
#     peaq_score = proc_speech_power / (raw_speech_power + 1e-10)
#     return peaq_score


# def calculate_latency(pipeline_timestamps, webrtc_rtt=None):
#     """Calculate total latency (processing + network RTT)."""
#     if not pipeline_timestamps or len(pipeline_timestamps) < 2:
#         raise ValueError("Pipeline timestamps must include start and end times.")
    
#     processing_latency = pipeline_timestamps[-1] - pipeline_timestamps[0]
#     network_latency = webrtc_rtt if webrtc_rtt is not None else 0.0
#     total_latency = processing_latency + (network_latency / 1000.0)  # Convert RTT (ms) to seconds
#     return {
#         "processing_latency": float(processing_latency),
#         "network_latency": float(network_latency / 1000.0),
#         "total_latency": float(total_latency)
#     }

# def log_mos_data(clarity=0, noise_reduction=0, level_matching=0):
#     """Structure MOS survey data for Flask integration."""
#     mos_data = {
#         "clarity": clarity,
#         "noise_reduction": noise_reduction,
#         "level_matching": level_matching,
#         "mos_average": (clarity + noise_reduction + level_matching) / 3.0
#     }
#     return mos_data

# def evaluate_audio(raw_path, processed_path, output_path, pipeline_timestamps=None, webrtc_rtt=None, sample_rate=48000):
#     """Evaluate audio quality using objective and subjective metrics."""
#     if not os.path.exists(raw_path):
#         raise FileNotFoundError(f"Raw audio file {raw_path} not found")
#     if not os.path.exists(processed_path):
#         raise FileNotFoundError(f"Processed audio file {processed_path} not found")
    
#     # Read audio files
#     sample_rate, raw_audio = wavfile.read(raw_path)
#     _, processed_audio = wavfile.read(processed_path)
    
#     # Ensure same length and mono
#     min_len = min(len(raw_audio), len(processed_audio))
#     raw_audio = raw_audio[:min_len]
#     processed_audio = processed_audio[:min_len]
#     if len(raw_audio.shape) > 1:
#         raw_audio = np.mean(raw_audio, axis=1)
#     if len(processed_audio.shape) > 1:
#         processed_audio = np.mean(processed_audio, axis=1)
    
#     # Calculate objective metrics
#     snr = calculate_snr(raw_audio, processed_audio)
    
#     peaq = calculate_peaq_approximation(raw_audio, processed_audio, sample_rate)
    
#     try:
#         latency = calculate_latency(pipeline_timestamps, webrtc_rtt)
#     except ValueError as e:
#         print(f"Latency error: {e}. Defaulting to zero.")
#         latency = {"processing_latency": 0.0, "network_latency": 0.0, "total_latency": 0.0}
    
#     # Placeholder MOS data (to be updated via Flask UI)
#     mos_data = log_mos_data(clarity=0, noise_reduction=0, level_matching=0)
    
#     # Compile results
#     results = {
#         "snr_db": float(snr),
#         "latency": latency,
#         "peaq_score": float(peaq),
#         # "peaq_source": peaq_source,
#         "mos": mos_data
#     }
    
#     # Save results to JSON
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=4)
    
#     return results

# if __name__ == "__main__":
#     # Example usage for testing
#     raw_audio = "uploads/test_audio.wav"
#     processed_audio = "static/processed/postprocessed_audio.wav"
#     output_results = "static/processed/evaluation_results.json"
#     # Simulated pipeline timestamps and WebRTC RTT
#     pipeline_timestamps = [time.time(), time.time() + 0.3]  # Example: 0.3s processing
#     webrtc_rtt = 50.0  # Example: 50ms RTT
#     try:
#         results = evaluate_audio(raw_audio, processed_audio, output_results, pipeline_timestamps, webrtc_rtt)
#         print(f"Evaluation results saved to {output_results}")
#         print(json.dumps(results, indent=4))
#     except Exception as e:
#         print(f"Error during evaluation: {e}")

import os
import numpy as np
import json
import time
import warnings
from scipy import signal
from scipy.io import wavfile

def calculate_snr(raw_audio, processed_audio):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    signal_power = np.mean(raw_audio ** 2)
    noise = raw_audio - processed_audio
    noise_power = np.mean(noise ** 2) + 1e-10  # Avoid division by zero
    print(f"[SNR DEBUG] Signal: {signal_power:.6f}, Noise: {noise_power:.6f}")

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_peaq_approximation(raw_audio, processed_audio, sample_rate):
    """Approximate Perceptual Evaluation of Audio Quality (PEAQ) using frequency analysis."""
    # Compute power spectral density for raw and processed audio
    freqs_raw, psd_raw = signal.welch(raw_audio, fs=sample_rate, nperseg=1024)
    freqs_proc, psd_proc = signal.welch(processed_audio, fs=sample_rate, nperseg=1024)
    
    # Focus on speech frequencies (300Hz - 3400Hz)
    speech_mask = (freqs_raw >= 300) & (freqs_raw <= 3400)
    raw_speech_power = np.mean(psd_raw[speech_mask])
    proc_speech_power = np.mean(psd_proc[speech_mask])
    
    # Simplified PEAQ: Ratio of speech power retention (higher is better)
    peaq_score = 10 * np.log10(proc_speech_power / (raw_speech_power + 1e-10))
    return peaq_score

def calculate_latency(pipeline_timestamps, webrtc_rtt=None):
    """Calculate total latency (processing + network RTT)."""
    if not pipeline_timestamps or len(pipeline_timestamps) < 2:
        raise ValueError("Pipeline timestamps must include start and end times.")
    
    processing_latency = pipeline_timestamps[-1] - pipeline_timestamps[0]
    network_latency = webrtc_rtt if webrtc_rtt is not None else 0.0
    total_latency = processing_latency + (network_latency / 1000.0)  # Convert RTT (ms) to seconds
    return {
        "processing_latency": float(processing_latency),
        "network_latency": float(network_latency / 1000.0),
        "total_latency": float(total_latency)
    }

def log_mos_data(clarity=0, noise_reduction=0, level_matching=0):
    """Structure MOS survey data for Flask integration."""
    mos_data = {
        "clarity": clarity,
        "noise_reduction": noise_reduction,
        "level_matching": level_matching,
        "mos_average": (clarity + noise_reduction + level_matching) / 3.0
    }
    return mos_data

def classify_snr(snr_db):
    if snr_db < 10:
        return "Unreliable"
    elif snr_db < 25:
        return "Poor"
    elif snr_db < 40:
        return "Good"
    else:
        return "Excellent"

def evaluate_audio(raw_path, processed_path, output_path, pipeline_timestamps=None, webrtc_rtt=None, sample_rate=16000):
    """Evaluate audio quality using objective and subjective metrics."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw audio file {raw_path} not found")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed audio file {processed_path} not found")
    
    # Read audio files
    try:
        raw_sample_rate, raw_audio = wavfile.read(raw_path)
        proc_sample_rate, processed_audio = wavfile.read(processed_path)
    except Exception as e:
        raise RuntimeError(f"Error reading audio files: {e}")
    
    # Handle different sample rates - resample if needed
    if raw_sample_rate != proc_sample_rate:
        warnings.warn(f"Sample rate mismatch: raw={raw_sample_rate}, processed={proc_sample_rate}. Resampling raw audio.")
        # Resample raw audio to match processed audio sample rate
        target_length = int(len(raw_audio) * proc_sample_rate / raw_sample_rate)
        raw_audio = signal.resample(raw_audio, target_length)
        raw_sample_rate = proc_sample_rate
    
    # Use the actual sample rate from the files
    actual_sample_rate = proc_sample_rate
    
    # Ensure same length and mono
    min_len = min(len(raw_audio), len(processed_audio))
    raw_audio = raw_audio[:min_len]
    processed_audio = processed_audio[:min_len]
    
    # Convert to mono if stereo
    if len(raw_audio.shape) > 1:
        raw_audio = np.mean(raw_audio, axis=1)
    if len(processed_audio.shape) > 1:
        processed_audio = np.mean(processed_audio, axis=1)
    
    # Ensure both arrays are float64 for calculations
    raw_audio = raw_audio.astype(np.float64)
    processed_audio = processed_audio.astype(np.float64)
    
    # # Normalize audio to prevent overflow in calculations
    # raw_audio = raw_audio / np.max(np.abs(raw_audio) + 1e-10)
    # processed_audio = processed_audio / np.max(np.abs(processed_audio) + 1e-10)
    # Normalize audio to prevent overflow
    denom_raw = np.max(np.abs(raw_audio))
    if denom_raw > 0:
        raw_audio = raw_audio / denom_raw

    denom_proc = np.max(np.abs(processed_audio))
    if denom_proc > 0:
        processed_audio = processed_audio / denom_proc

    
    # Calculate objective metrics
    try:
        snr = calculate_snr(raw_audio, processed_audio)
    except Exception as e:
        warnings.warn(f"SNR calculation failed: {e}. Setting to 0.")
        snr = 0.0
    
    try:
        peaq = calculate_peaq_approximation(raw_audio, processed_audio, actual_sample_rate)
    except Exception as e:
        warnings.warn(f"PEAQ calculation failed: {e}. Setting to 0.")
        peaq = 0.0
    
    # Calculate latency (processing + network)
    try:
        latency = calculate_latency(pipeline_timestamps, webrtc_rtt)
    except ValueError as e:
        print(f"Latency error: {e}. Defaulting to zero.")
        latency = {"processing_latency": 0.0, "network_latency": 0.0, "total_latency": 0.0}
    
    # Placeholder MOS data (to be updated via Flask UI)
    mos_data = log_mos_data(clarity=0, noise_reduction=0, level_matching=0)
    
    # Compile results
    results = {
        "snr_db": float(snr),
        "snr_class": classify_snr(snr),
        "latency": latency,
        "peaq_score": float(peaq),
        "mos": mos_data,
        "audio_info": {
            "sample_rate": int(actual_sample_rate),
            "duration_seconds": float(len(processed_audio) / actual_sample_rate),
            "audio_length_samples": int(len(processed_audio))
        }
    }
    
    # Save results to JSON
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving results to {output_path}: {e}")
    
    return results

if __name__ == "__main__":
    # Example usage for testing
    raw_audio = "uploads/test_audio.wav"
    processed_audio = "static/processed/postprocessed_audio.wav"
    output_results = "static/processed/evaluation_results.json"
    
    # Simulated pipeline timestamps and WebRTC RTT
    pipeline_timestamps = [time.time(), time.time() + 0.3]  # Example: 0.3s processing
    webrtc_rtt = 50.0  # Example: 50ms RTT
    
    try:
        results = evaluate_audio(raw_audio, processed_audio, output_results, pipeline_timestamps, webrtc_rtt)
        print(f"Evaluation results saved to {output_results}")
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Error during evaluation: {e}")