
# import os
# import sys
# from pydub import AudioSegment
# import subprocess

# # Absolute path to the folder containing libopus.dll
# dll_dir = os.path.abspath(os.path.dirname(__file__))

# # Add the DLL folder to search path for dynamic libraries
# if sys.version_info >= (3, 8):
#     os.add_dll_directory(dll_dir)
# else:
#     os.environ["PATH"] += os.pathsep + dll_dir


# from opuslib import Encoder
# import numpy as np

# # def encode_to_opus(input_path, output_path, sample_rate=48000, bitrate=24000):
# #     """Encode preprocessed audio to Opus format."""
# #     if not os.path.exists(input_path):
# #         raise FileNotFoundError(f"Input audio file {input_path} not found")
    
# #     # Load preprocessed audio
# #     audio = AudioSegment.from_wav(input_path)
    
# #     # Ensure mono and correct sample rate
# #     audio = audio.set_channels(1).set_frame_rate(sample_rate)
    
# #     # Convert audio to raw PCM data
# #     raw_data = np.frombuffer(audio.raw_data, dtype=np.int16)
    
# #     # Initialize Opus encoder
# #     encoder = Encoder(sample_rate, 1, 'audio')  # Mono, audio application
# #     encoder.bitrate = bitrate  # Set bitrate (e.g., 24 kbps for speech)
    
# #     # Encode audio in chunks (20ms frames, standard for Opus)
# #     frame_size = int(sample_rate * 0.02)  # 20ms at sample_rate
# #     encoded_data = bytearray()
    
# #     for i in range(0, len(raw_data), frame_size):
# #         frame = raw_data[i:i + frame_size]
# #         if len(frame) < frame_size:
# #             # Pad with zeros if frame is too short
# #             frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
# #         encoded_frame = encoder.encode(frame.tobytes(), frame_size)
# #         encoded_data.extend(encoded_frame)
    
# #     # Save encoded audio to .opus file
# #     with open(output_path, 'wb') as f:
# #         f.write(encoded_data)
    
# #     return output_path


# # def encode_to_opus(input_path, output_path, sample_rate=16000, bitrate=24000):
# #     """Encode preprocessed audio to Opus format in OGG container."""
# #     if not os.path.exists(input_path):
# #         raise FileNotFoundError(f"Input audio file {input_path} not found")
    
# #     # Use FFmpeg to encode to Opus in OGG container
# #     AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"
    
# #     try:
# #         # Load preprocessed audio
# #         audio = AudioSegment.from_wav(input_path)
        
# #         # Ensure mono and correct sample rate
# #         audio = audio.set_channels(1).set_frame_rate(sample_rate)
        
# #         # Export as Opus in OGG container
# #         audio.export(
# #             output_path,
# #             format="ogg",
# #             codec="libopus",
# #             parameters=["-b:a", f"{bitrate}"]
# #         )
        
# #         return output_path
        
# #     except Exception as e:
# #         print(f"Opus encoding error: {e}")
# #         raise




# # def encode_to_opus(input_path, output_path, sample_rate=16000, bitrate=24000):
# #     """Encode preprocessed audio to Opus format in OGG container."""
# #     if not os.path.exists(input_path):
# #         raise FileNotFoundError(f"Input audio file {input_path} not found")
    
# #     # Explicit FFmpeg binary path (customize for your system)
# #     AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"

# #     try:
# #         print(f"[ENCODE] Loading input file: {input_path}")
# #         # Support WAV, MP3, OGG, etc. based on extension or file header
# #         audio = AudioSegment.from_file(input_path)

# #         print(f"[ENCODE] Original channels: {audio.channels}, frame rate: {audio.frame_rate}")

# #         # Enforce mono and sample rate for Opus compatibility
# #         audio = audio.set_channels(1).set_frame_rate(sample_rate)

# #         # Safe normalization with proper clipping control
# #         # samples = np.array(audio.get_array_of_samples()).astype(np.float32)
# #         # peak = np.max(np.abs(samples))
# #         # if peak > 0:
# #         #     gain = (0.85 * 32767) / peak
# #         #     samples = np.clip(samples * gain, -32768, 32767)  # avoid overflow

# #         # samples_int16 = samples.astype(np.int16)

# #         # audio = AudioSegment(
# #         #     samples_int16.tobytes(),
# #         #     frame_rate=sample_rate,
# #         #     sample_width=2,
# #         #     channels=1
# #         # )

# #         # # Ensure bitrate format is correct for ffmpeg
# #         # if isinstance(bitrate, int):
# #         #     bitrate_param = f"{int(bitrate / 1000)}k"
# #         # else:
# #         #     bitrate_param = bitrate
        
# #         # Avoid aggressive manual normalization
# #         target_dBFS = -18.0
# #         change_in_dBFS = target_dBFS - audio.dBFS
# #         audio = audio.apply_gain(change_in_dBFS)

# #         # Export using libopus codec
# #         bitrate_param = f"{int(bitrate / 1000)}k" if isinstance(bitrate, int) else bitrate


# #         audio.export(
# #             output_path,
# #             format="ogg",
# #             codec="libopus",
# #             parameters=["-b:a", bitrate_param]
# #         )

# #         print(f"[ENCODE] Export successful: {output_path}")
# #         return output_path

# #     except Exception as e:
# #         print(f"[ENCODE ERROR] Opus encoding error: {e}")
# #         raise



# def encode_to_opus(input_path, output_path, sample_rate=48000, bitrate=96000, normalize=True):
#     """Encode audio to Opus format with flexible options."""
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input audio file {input_path} not found")
    
#     AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"

#     try:
#         print(f"[ENCODE] Loading input file: {input_path}")
#         audio = AudioSegment.from_file(input_path)
#         print(f"[ENCODE] Original: channels={audio.channels}, frame_rate={audio.frame_rate}, duration={audio.duration_seconds}s")

#         audio = audio.set_channels(1).set_frame_rate(sample_rate)
#         print(f"[ENCODE] After processing: channels={audio.channels}, frame_rate={audio.frame_rate}")

#         if normalize:
#             samples = np.array(audio.get_array_of_samples()).astype(np.float32)
#             peak = np.max(np.abs(samples))
#             print(f"[ENCODE] Peak amplitude: {peak}")
#             if peak > 0:
#                 gain = (0.8 * 32767) / peak
#                 samples = np.clip(samples * gain, -32768, 32767)
#                 samples += np.random.normal(0, 0.5, samples.shape)  # Dither
#                 print(f"[ENCODE] Applied gain: {gain}")
#             samples_int16 = samples.astype(np.int16)
#             audio = AudioSegment(
#                 samples_int16.tobytes(),
#                 frame_rate=sample_rate,
#                 sample_width=2,
#                 channels=1
#             )

#         bitrate_param = f"{int(bitrate / 1000)}k" if isinstance(bitrate, int) else bitrate
#         audio.export(
#             output_path,
#             format="ogg",
#             codec="libopus",
#             parameters=["-b:a", bitrate_param, "-vbr", "on", "-application", "audio"]
#         )

#         print(f"[ENCODE] Export successful: {output_path}")
#         return output_path

#     except Exception as e:
#         print(f"[ENCODE ERROR] Opus encoding error: {e}")
#         raise

# if __name__ == "__main__":
#     input_audio = "static/processed/preprocessed_audio.wav"
#     output_audio = "static/processed/encoded_audio.wav"

#     try:
#         # Test 1: Baseline (no normalization, 48kHz, 96kbps)
#         encode_to_opus(input_audio, output_audio, sample_rate=48000, bitrate="96k", normalize=False)
#         print(f"[TEST 1 COMPLETE] Encoded audio saved to {output_audio}")
#         # Check for crackling at 7s, 28s, 44s

#         # Test 2: Normalization enabled (48kHz, 96kbps)
#         # encode_to_opus(input_audio, output_audio, sample_rate=48000, bitrate="96k", normalize=True)
#         # print(f"[TEST 2 COMPLETE] Encoded audio saved to {output_audio}")
#         # Check for crackling at 7s, 28s, 44s

#         # Test 3: Higher bitrate (48kHz, 128kbps, normalization)
#         # encode_to_opus(input_audio, output_audio, sample_rate=48000, bitrate="128k", normalize=True)
#         # print(f"[TEST 3 COMPLETE] Encoded audio saved to {output_audio}")
#         # Check for crackling at 7s, 28s, 44s

#         # Test 4: Lower sample rate (24kHz, 96kbps, normalization)
#         # encode_to_opus(input_audio, output_audio, sample_rate=24000, bitrate="96k", normalize=True)
#         # print(f"[TEST 4 COMPLETE] Encoded audio saved to {output_audio}")
#         # Check for crackling at 7s, 28s, 44s

#         # Test 5: Direct FFmpeg (run in terminal, not Python)
#         # Command: ffmpeg -i C:\Users\lovef\Downloads\Daniel.ogg -c:a libopus -b:a 96k -vbr on -application audio -ar 48000 -ac 1 C:\Users\lovef\Downloads\encoded_audio_test5.ogg
#         # print("[TEST 5] Run FFmpeg command in terminal and check output")
#         # Check for crackling at 7s, 28s, 44s

#     except Exception as e:
#         print(f"[TEST FAILED] Error during Opus encoding: {e}")

# # if __name__ == "__main__":
# #     # Example usage for testing
# #     # input_audio = "static/processed/preprocessed_audio.wav"
# #     output_audio = "static/processed/encoded_audio.ogg"
# #     input_audio = "uploads/test_audio.ogg"

#     # try:
#     #     # Test 1: Full encoding path (standard)
#     #     # encode_to_opus(input_audio, output_audio, sample_rate=16000, bitrate="32k")

#     #     encode_to_opus(input_audio, output_audio)

#     #     # Test 2: Int bitrate handling
#     #     # encode_to_opus(input_audio, output_audio, bitrate=32000)

#     #     # Test 3: Custom sample rate (may produce warnings)
#     #     # encode_to_opus(input_audio, output_audio, sample_rate=8000, bitrate="16k")

#     #     print(f"[TEST COMPLETE] Encoded audio saved to {output_audio}")

#     # except Exception as e:
#     #     print(f"[TEST FAILED] Error during Opus encoding: {e}")






import os
from pydub import AudioSegment
import numpy as np

# Optional: Set FFmpeg path if not globally available
AudioSegment.ffmpeg = "C:\\Program Files\\ffmpeg\\ffmpegbuild\\bin\\ffmpeg.exe"

def normalize_audio(audio, sample_rate):
    """Normalize audio to safe peak using dithered gain."""
    try:
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        peak = np.max(np.abs(samples))
        print(f"[NORMALIZE] Peak amplitude: {peak}")

        if peak > 0:
            gain = (0.8 * 32767) / peak
            samples = np.clip(samples * gain, -32768, 32767)
            samples += np.random.normal(0, 0.5, samples.shape)  # Dither
            print(f"[NORMALIZE] Applied gain: {gain:.2f}")

        samples_int16 = samples.astype(np.int16)
        return AudioSegment(
            samples_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}")
        return audio


def encode_to_opus(input_path, output_path,
                   sample_rate=48000,
                   bitrate="96k",
                   apply_normalization=True):
    """Encode audio to Opus format using FFmpeg via PyDub."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        print(f"[ENCODE] Loading input: {input_path}")
        audio = AudioSegment.from_file(input_path)

        print(f"[INFO] Original: {audio.channels}ch @ {audio.frame_rate} Hz, {audio.duration_seconds:.2f}s")

        # Force mono and desired sample rate
        audio = audio.set_channels(1).set_frame_rate(sample_rate)

        if apply_normalization:
            audio = normalize_audio(audio, sample_rate)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audio.export(
            output_path,
            format="ogg",
            codec="libopus",
            parameters=["-b:a", bitrate, "-vbr", "on", "-application", "audio"]
        )

        print(f"[SUCCESS] Encoded Opus file saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"[ENCODE ERROR] Failed to encode with Opus: {e}")
        raise


if __name__ == "__main__":
    input_audio = "static/processed/preprocessed_audio.wav"
    output_audio = "static/processed/encoded_audio.ogg"

    try:
        encode_to_opus(
            input_path=input_audio,
            output_path=output_audio,
            sample_rate=48000,     # WebRTC standard
            bitrate="96k",
            apply_normalization=False  # Set to False for baseline/control
        )
    except Exception as err:
        print(f"[TEST FAILED] Encoding error: {err}")
