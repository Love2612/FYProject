import os
import numpy as np
from scipy import signal
from pydub import AudioSegment
# from utils.rnnoise_wrapper import RNNoise  # Assuming RNNoise Python bindings are available
import webrtcvad
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
from scipy.signal import stft, istft, butter, lfilter


def decode_opus(input_path):
    """Decode Opus audio to PCM using pydub with explicit FFmpeg path."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file {input_path} not found")
    
    # Explicitly set FFmpeg path (update to your FFmpeg location)
    AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"  
    
    try:
        # Load and decode Opus file to WAV
        audio = AudioSegment.from_file(input_path, format="ogg", codec="libopus")
        audio = audio.set_channels(1).set_frame_rate(16000)  # Ensure mono, 48kHz
        # raw_data = np.frombuffer(audio.raw_data, dtype=np.int16)
        # return raw_data, 16000
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        return samples, audio.frame_rate
    except Exception as e:
        print(f"FFmpeg decoding error: {e}")
        raise

# def rnnoise_denoise(audio_data, sample_rate):
#     """Apply RNNoise for machine-learning-based noise suppression."""
#     try:
#         # Initialize RNNoise
#         denoiser = rnnoise.RNNoise()
        
#         # Process audio in 10ms frames (RNNoise expects 10ms at 48kHz = 480 samples)
#         frame_size = int(sample_rate * 0.01)
#         denoised_data = []
        
#         for i in range(0, len(audio_data), frame_size):
#             frame = audio_data[i:i + frame_size]
#             if len(frame) < frame_size:
#                 frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
#             denoised_frame = denoiser.process_frame(frame.astype(np.float32))
#             denoised_data.extend(denoised_frame)
        
#         return np.array(denoised_data, dtype=np.int16)
#     except Exception as e:
#         warnings.warn(f"RNNoise processing failed: {e}. Returning original audio.")
#         return audio_data

# def rnnoise_denoise(audio_data, sample_rate):
#     """Apply RNNoise for machine-learning-based noise suppression."""
#     try:
#         denoiser = RNNoise(sample_rate)
#         frame_size = int(sample_rate * 0.01)  # 10ms frames
#         denoised_data = []
        
#         for i in range(0, len(audio_data), frame_size):
#             frame = audio_data[i:i + frame_size]
#             if len(frame) < frame_size:
#                 frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
#             denoised_frame = denoiser.process_frame(frame)
#             denoised_data.extend(denoised_frame)
        
#         return np.array(denoised_data, dtype=np.int16)
#     except Exception as e:
#         warnings.warn(f"RNNoise processing failed: {e}. Returning original audio.")
#         return audio_data





# def vad_denoise(audio_data, sample_rate):
#     """Apply noise suppression using WebRTC VAD."""
#     try:
#         # Initialize WebRTC VAD
#         vad = webrtcvad.Vad()
#         vad.set_mode(3)  # Aggressive mode for noise suppression
        
#         # Process audio in 10ms frames (480 samples at 48kHz)
#         frame_size = int(sample_rate * 0.01)  # 10ms frames
#         denoised_data = np.copy(audio_data).astype(np.int16)
        
#         for i in range(0, len(audio_data), frame_size):
#             frame = audio_data[i:i + frame_size]
#             if len(frame) < frame_size:
#                 frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
            
#             # Convert frame to bytes for VAD
#             frame_bytes = frame.astype(np.int16).tobytes()
            
#             # Check if frame contains speech
#             is_speech = vad.is_speech(frame_bytes, sample_rate)
            
#             # Suppress non-speech frames (set to zero or reduce amplitude)
#             if not is_speech:
#                 denoised_data[i:i + frame_size] = frame * 0.1  # Reduce amplitude by 90%
        
#         return denoised_data
#     except Exception as e:
#         warnings.warn(f"VAD processing failed: {e}. Returning original audio.")
#         return audio_data

# def dynamic_range_compression(audio_data, sample_rate, threshold=-20.0, ratio=4.0):
#     """Apply dynamic range compression to balance volume."""
#     # Convert audio to dB
#     audio_db = 20 * np.log10(np.abs(audio_data) + 1e-10)
    
#     # Apply compression where audio exceeds threshold
#     gain = np.where(audio_db > threshold, threshold + (audio_db - threshold) / ratio, audio_db)
#     gain = gain - audio_db  # Calculate gain reduction
    
#     # Apply gain to original signal
#     compressed_data = audio_data * np.power(10, gain / 20.0)
#     return compressed_data.astype(np.int16)

# def echo_cancellation(audio_data, sample_rate, reference_data=None):
#     """Apply basic acoustic echo cancellation (simplified)."""
#     if reference_data is None:
#         # If no reference signal, assume minimal echo or return original
#         warnings.warn("No reference signal provided for AEC. Returning original audio.")
#         return audio_data
    
#     # Simple AEC: Cross-correlation to estimate delay, then subtract
#     correlation = signal.correlate(audio_data, reference_data, mode='full')
#     delay = np.argmax(correlation) - len(reference_data) + 1
    
#     if delay >= 0:
#         reference_data = np.pad(reference_data, (delay, 0), mode='constant')
#         reference_data = reference_data[:len(audio_data)]
#     else:
#         reference_data = np.pad(reference_data, (0, -delay), mode='constant')
#         reference_data = reference_data[-delay:len(audio_data) - delay]
    
#     # Subtract reference (scaled) to remove echo
#     scaling_factor = 0.8  # Empirical scaling to avoid over-subtraction
#     clean_data = audio_data - (scaling_factor * reference_data)
#     return clean_data.astype(np.int16)

# def postprocess_audio(input_path, output_path, reference_path=None, sample_rate=48000):
#     """Main function to post-process audio with de-noising, DRC, and AEC."""
#     # Decode Opus audio
#     audio_data, sample_rate = decode_opus(input_path)
    
#     # Load reference audio for AEC if provided
#     reference_data = None
#     if reference_path and os.path.exists(reference_path):
#         reference_audio = AudioSegment.from_wav(reference_path)
#         reference_audio = reference_audio.set_channels(1).set_frame_rate(sample_rate)
#         reference_data = np.frombuffer(reference_audio.raw_data, dtype=np.int16)
    
#     # Step 1: De-noising with RNNoise
#     audio_denoised = vad_denoise(audio_data, sample_rate)
    
#     # Step 2: Dynamic range compression
#     audio_compressed = dynamic_range_compression(audio_denoised, sample_rate)
    
#     # Step 3: Echo cancellation
#     audio_processed = echo_cancellation(audio_compressed, sample_rate, reference_data)
    
#     # Save processed audio as WAV
#     output_audio = AudioSegment(
#         audio_processed.tobytes(),
#         frame_rate=sample_rate,
#         sample_width=2,
#         channels=1
#     )
#     output_audio.export(output_path, format="wav")
#     return output_path

# if __name__ == "__main__":
#     # Example usage for testing
#     input_audio = "static/processed/encoded_audio.ogg"
#     output_audio = "static/processed/postprocessed_audio.wav"
#     reference_audio = None  # Replace with path to reference audio if available
#     try:
#         postprocess_audio(input_audio, output_audio, reference_audio)
#         print(f"Post-processed audio saved to {output_audio}")
#     except Exception as e:
#         print(f"Error during post-processing: {e}")
        




def vad_denoise(audio_data, sample_rate):
    """Apply noise suppression using WebRTC VAD."""
    try:
        # Initialize WebRTC VAD
        vad = webrtcvad.Vad()
        vad.set_mode(2)  # Aggressive mode for noise suppression
        
        # WebRTC VAD frame size requirements for 16kHz:
        # 10ms=160, 20ms=320, 30ms=480 samples
        frame_size = 480  # 30ms frames for 16kHz
        
        # Ensure sample rate is 16kHz
        if sample_rate != 16000:
            # Resample to 16kHz
            target_length = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, target_length).astype(np.int16)
            sample_rate = 16000
        
        # Pad audio to be divisible by frame_size to avoid broadcast errors
        remainder = len(audio_data) % frame_size
        if remainder != 0:
            padding = frame_size - remainder
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        denoised_data = np.copy(audio_data).astype(np.int16)
        
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            
            # Convert frame to bytes for VAD (ensure int16)
            frame_bytes = frame.astype(np.int16).tobytes()
            
            # Check if frame contains speech
            is_speech = vad.is_speech(frame_bytes, sample_rate)
            
            # Suppress non-speech frames (reduce amplitude by 90%)
            if not is_speech:
                denoised_data[i:i + frame_size] = frame * 0.4
        
        return denoised_data
    except Exception as e:
        warnings.warn(f"VAD processing failed: {e}. Returning original audio.")
        return audio_data

def dynamic_range_compression(audio_data, sample_rate, threshold=-20.0, ratio=4.0):
    """Apply dynamic range compression to balance volume."""
    # Convert audio to dB
    audio_db = 20 * np.log10(np.abs(audio_data) + 1e-10)
    
    # Apply compression where audio exceeds threshold
    gain = np.where(audio_db > threshold, threshold + (audio_db - threshold) / ratio, audio_db)
    gain = gain - audio_db  # Calculate gain reduction
    
    # Apply gain to original signal
    compressed_data = audio_data * np.power(10, gain / 20.0)
    return compressed_data.astype(np.int16)

def echo_cancellation(audio_data, sample_rate, reference_data=None):
    """Apply basic acoustic echo cancellation (simplified)."""
    if reference_data is None:
        # Return original audio without warning for cleaner output
        return audio_data
    
    # Simple AEC: Cross-correlation to estimate delay, then subtract
    correlation = signal.correlate(audio_data, reference_data, mode='full')
    delay = np.argmax(correlation) - len(reference_data) + 1
    
    if delay >= 0:
        reference_data = np.pad(reference_data, (delay, 0), mode='constant')
        reference_data = reference_data[:len(audio_data)]
    else:
        reference_data = np.pad(reference_data, (0, -delay), mode='constant')
        reference_data = reference_data[-delay:len(audio_data) - delay]
    
    # Subtract reference (scaled) to remove echo
    scaling_factor = 0.8  # Empirical scaling to avoid over-subtraction
    clean_data = audio_data - (scaling_factor * reference_data)
    return clean_data.astype(np.int16)

def presence_boost(audio_data, sample_rate):
    """Boost 3–5kHz range to improve speech clarity and presence."""
    nyq = sample_rate / 2
    low = 3000 / nyq
    high = 5000 / nyq
    b, a = butter(2, [low, high], btype='band')
    boost = lfilter(b, a, audio_data)
    return np.clip(audio_data + 0.4 * boost, -32768, 32767)


def postprocess_audio(input_path, output_path, reference_path=None, sample_rate=16000):
    """Main function to post-process audio with de-noising, DRC, and AEC."""
    # Decode Opus audio
    audio_data, sample_rate = decode_opus(input_path)
    
    # Load reference audio for AEC if provided
    reference_data = None
    if reference_path and os.path.exists(reference_path):
        reference_audio = AudioSegment.from_wav(reference_path)
        reference_audio = reference_audio.set_channels(1).set_frame_rate(sample_rate)
        reference_data = np.frombuffer(reference_audio.raw_data, dtype=np.int16)
    
    # Step 1: De-noising with VAD
    audio_denoised = vad_denoise(audio_data, sample_rate)
    
    # Step 2: Dynamic range compression
    audio_compressed = dynamic_range_compression(audio_denoised, sample_rate)
    
    # Step 3: Echo cancellation
    # audio_processed = echo_cancellation(audio_compressed, sample_rate, reference_data)
    audio_processed = echo_cancellation(audio_compressed, sample_rate, reference_data)

    # [NEW] Presence boost to recover speech brightness
    audio_processed = presence_boost(audio_processed, sample_rate)

    # Trim or match to expected length
    expected_samples = int(sample_rate * (len(audio_data) / sample_rate))
    audio_processed = audio_processed[:expected_samples]

    # Normalize final signal if it's too quiet
    peak = np.max(np.abs(audio_processed))
    if peak < 5000:
        print(f"[DEBUG] Audio signal too quiet (peak={peak}), applying normalization.")
        audio_processed = audio_processed * (32767 / (peak + 1e-10))
    
    audio_processed = np.clip(audio_processed, -32768, 32767)  # Ensure it's in int16 range

    print(f"[DEBUG] Processed duration: {len(audio_processed)/sample_rate:.2f}s")
    print(f"[DEBUG] Input duration: {len(audio_data)/sample_rate:.2f}s")

    
    # Save processed audio as WAV
    output_audio = AudioSegment(
        audio_processed.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

    output_audio.export(output_path, format="wav")
    return output_path


if __name__ == "__main__":
    # Example usage for testing
    input_audio = "static/processed/encoded_audio.ogg"
    output_audio = "static/processed/postprocessed_audio.wav"
    reference_audio = None  # Replace with path to reference audio if available
    try:
        postprocess_audio(input_audio, output_audio, reference_audio)
        print(f"Post-processed audio saved to {output_audio}")
    except Exception as e:
        print(f"Error during post-processing: {e}")