import os
import sys
from pydub import AudioSegment

# Absolute path to the folder containing libopus.dll
dll_dir = os.path.abspath(os.path.dirname(__file__))

# Add the DLL folder to search path for dynamic libraries
if sys.version_info >= (3, 8):
    os.add_dll_directory(dll_dir)
else:
    os.environ["PATH"] += os.pathsep + dll_dir


from opuslib import Encoder
import numpy as np

# def encode_to_opus(input_path, output_path, sample_rate=48000, bitrate=24000):
#     """Encode preprocessed audio to Opus format."""
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input audio file {input_path} not found")
    
#     # Load preprocessed audio
#     audio = AudioSegment.from_wav(input_path)
    
#     # Ensure mono and correct sample rate
#     audio = audio.set_channels(1).set_frame_rate(sample_rate)
    
#     # Convert audio to raw PCM data
#     raw_data = np.frombuffer(audio.raw_data, dtype=np.int16)
    
#     # Initialize Opus encoder
#     encoder = Encoder(sample_rate, 1, 'audio')  # Mono, audio application
#     encoder.bitrate = bitrate  # Set bitrate (e.g., 24 kbps for speech)
    
#     # Encode audio in chunks (20ms frames, standard for Opus)
#     frame_size = int(sample_rate * 0.02)  # 20ms at sample_rate
#     encoded_data = bytearray()
    
#     for i in range(0, len(raw_data), frame_size):
#         frame = raw_data[i:i + frame_size]
#         if len(frame) < frame_size:
#             # Pad with zeros if frame is too short
#             frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
#         encoded_frame = encoder.encode(frame.tobytes(), frame_size)
#         encoded_data.extend(encoded_frame)
    
#     # Save encoded audio to .opus file
#     with open(output_path, 'wb') as f:
#         f.write(encoded_data)
    
#     return output_path


# def encode_to_opus(input_path, output_path, sample_rate=16000, bitrate=24000):
#     """Encode preprocessed audio to Opus format in OGG container."""
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input audio file {input_path} not found")
    
#     # Use FFmpeg to encode to Opus in OGG container
#     AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"
    
#     try:
#         # Load preprocessed audio
#         audio = AudioSegment.from_wav(input_path)
        
#         # Ensure mono and correct sample rate
#         audio = audio.set_channels(1).set_frame_rate(sample_rate)
        
#         # Export as Opus in OGG container
#         audio.export(
#             output_path,
#             format="ogg",
#             codec="libopus",
#             parameters=["-b:a", f"{bitrate}"]
#         )
        
#         return output_path
        
#     except Exception as e:
#         print(f"Opus encoding error: {e}")
#         raise

def encode_to_opus(input_path, output_path, sample_rate=16000, bitrate=24000):
    """Encode preprocessed audio to Opus format in OGG container."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file {input_path} not found")
    
    # Use FFmpeg to encode to Opus in OGG container
    AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"
    
    try:
        # Load preprocessed audio
        audio = AudioSegment.from_wav(input_path)
        
        # Ensure mono and correct sample rate
        audio = audio.set_channels(1).set_frame_rate(sample_rate)
        
        # Export as Opus in OGG container
        # Ensure bitrate is a string like '32k'
        bitrate_param = f"{bitrate}k" if isinstance(bitrate, int) else bitrate
        audio.export(
            output_path,
            format="ogg",
            codec="libopus",
            parameters=["-b:a", bitrate_param]
        )

        
        return output_path
        
    except Exception as e:
        print(f"Opus encoding error: {e}")
        raise

if __name__ == "__main__":
    # Example usage for testing
    input_audio = "static/processed/preprocessed_audio.wav"
    output_audio = "static/processed/encoded_audio.ogg"
    try:
        encode_to_opus(input_audio, output_audio)
        print(f"Encoded audio saved to {output_audio}")
    except Exception as e:
        print(f"Error during Opus encoding: {e}")   