#!/usr/bin/env python3
"""
Helper script to transcribe your voice sample for use with voice_clone.py
"""

import argparse
import sys

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Run: uv sync")
    sys.exit(1)


def transcribe_audio(audio_path, model_size="base", device="cuda"):
    """
    Transcribe an audio file using Whisper

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        device: Device to run on (cuda/cpu)

    Returns:
        Transcribed text
    """
    print(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(
        model_size, device=device, compute_type="float16" if device == "cuda" else "int8"
    )

    print(f"Transcribing {audio_path}...")
    segments, info = model.transcribe(audio_path, language="en", vad_filter=True)

    # Combine all segments
    text = " ".join([segment.text for segment in segments]).strip()

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe voice sample for use with voice_clone.py"
    )
    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    try:
        text = transcribe_audio(args.audio_file, args.model, args.device)

        print("\n" + "=" * 80)
        print("TRANSCRIPT:")
        print("=" * 80)
        print(text)
        print("=" * 80)
        print("\nUse this transcript with voice_clone.py:")
        print(f'uv run voice_clone.py {args.audio_file} --voice-text "{text}"')
        print()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
