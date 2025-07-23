
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
