#!/usr/bin/env python3
"""
Voice Cloning Virtual Audio Device
Routes microphone input through Qwen3-TTS with voice cloning for use in conferencing apps
Uses local Qwen3-TTS models - no data sent to cloud
"""

import argparse
import queue
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel
from scipy import signal

# Audio processing constants
WHISPER_SAMPLE_RATE = 16000  # Whisper uses 16kHz
DEFAULT_OUTPUT_SAMPLE_RATE = 24000  # Default TTS output sample rate
DEFAULT_LANGUAGE = "en"  # Default language for transcription

# TTS generation parameters
TTS_MAX_TOKENS = 2048
TTS_TOP_K = 50
TTS_TOP_P = 1.0
TTS_TEMPERATURE = 0.9
TTS_REPETITION_PENALTY = 1.05
TTS_LANGUAGE = "Auto"  # Auto-detect language for TTS

# Processing parameters
QUEUE_TIMEOUT = 0.1  # Timeout for queue operations (seconds)
MAIN_LOOP_SLEEP = 0.1  # Sleep duration in main loop (seconds)


class VoiceCloner:
    def __init__(
        self,
        voice_sample_path,
        voice_sample_text=None,
        whisper_model_size="base",
        tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device="cuda",
        compute_type=None,  # Will auto-detect based on device
        vad_filter=True,
        chunk_duration=3.0,
        x_vector_only_mode=False,
    ):
        """
        Initialize the voice cloner

        Args:
            voice_sample_path: Path to audio file containing voice to clone
            voice_sample_text: Transcript of the voice sample (optional if x_vector_only_mode=True)
            whisper_model_size: Whisper model size (tiny, base, small, medium, large)
            tts_model_name: Qwen3-TTS model name or path
            device: Device to run models on (cuda, cpu)
            compute_type: Computation type for Whisper
            vad_filter: Enable Voice Activity Detection
            chunk_duration: Duration of audio chunks to process (seconds)
            x_vector_only_mode: Use only speaker embedding (lower quality but no ref_text needed)
        """
        self.voice_sample_path = Path(voice_sample_path)
        self.voice_sample_text = voice_sample_text
        self.chunk_duration = chunk_duration
        self.sample_rate = WHISPER_SAMPLE_RATE
        self.x_vector_only_mode = x_vector_only_mode

        # Verify voice sample exists
        if not self.voice_sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")

        # Load voice sample to get output sample rate
        voice_audio, voice_sr = sf.read(str(self.voice_sample_path))
        print(f"Voice sample loaded: {self.voice_sample_path}")
        print(f"Voice sample: {len(voice_audio)} samples at {voice_sr}Hz")

        # Initialize Whisper for STT
        print(f"Loading Whisper model ({whisper_model_size})...")
        # Auto-detect compute type if not specified
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"
        print(f"Using compute type: {compute_type}")
        self.whisper_model = WhisperModel(
            whisper_model_size, device=device, compute_type=compute_type
        )
        self.vad_filter = vad_filter

        # Initialize local Qwen3-TTS for voice cloning
        print(f"Loading Qwen3-TTS model ({tts_model_name})...")
        # Don't use flash_attention_2 on CPU or if not available
        attn_impl = "sdpa" if device == "cpu" else "flash_attention_2"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        print(f"Using attention: {attn_impl}, dtype: {dtype}")
        self.tts_model = Qwen3TTSModel.from_pretrained(
            tts_model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        # Create voice clone prompt once and reuse it
        print("Creating voice clone prompt from sample...")
        self.voice_clone_prompt = self.tts_model.create_voice_clone_prompt(
            ref_audio=str(self.voice_sample_path),
            ref_text=self.voice_sample_text,
            x_vector_only_mode=self.x_vector_only_mode,
        )
        print("Voice clone prompt ready!")

        # Audio buffers
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Control flags
        self.running = False
        self.processing_thread = None

        # Output sample rate (from TTS model)
        self.output_sample_rate = None  # Will be set on first generation

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        # Add audio to processing queue
        self.input_queue.put(indata.copy())

    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio through STT -> TTS pipeline"""
        try:
            # Save audio chunk to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
                sf.write(tmp_input.name, audio_chunk, self.sample_rate)
                tmp_input_path = tmp_input.name

            # Speech-to-Text with Whisper
            segments, info = self.whisper_model.transcribe(
                tmp_input_path, vad_filter=self.vad_filter, language=DEFAULT_LANGUAGE
            )

            # Combine all segments into text
            text = " ".join([segment.text for segment in segments]).strip()

            # Clean up temp input file
            Path(tmp_input_path).unlink()

            if not text:
                print("No speech detected")
                return None

            print(f"Transcribed: {text}")

            # Text-to-Speech with voice cloning (local)
            wavs, sr = self.tts_model.generate_voice_clone(
                text=text,
                language=TTS_LANGUAGE,
                voice_clone_prompt=self.voice_clone_prompt,
                max_new_tokens=TTS_MAX_TOKENS,
                do_sample=True,
                top_k=TTS_TOP_K,
                top_p=TTS_TOP_P,
                temperature=TTS_TEMPERATURE,
                repetition_penalty=TTS_REPETITION_PENALTY,
            )

            # Set output sample rate on first generation
            if self.output_sample_rate is None:
                self.output_sample_rate = sr
                print(f"TTS output sample rate: {sr}Hz")

            if wavs and len(wavs) > 0:
                audio_data = wavs[0]  # Get first (and only) output
                print(f"Generated TTS audio: {len(audio_data)} samples at {sr}Hz")
                return audio_data, sr

        except Exception as e:
            print(f"Error processing audio chunk: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

        return None

    def processing_loop(self):
        """Main processing loop - runs in separate thread"""
        audio_buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.chunk_duration * self.sample_rate)

        print("Processing loop started")

        while self.running:
            try:
                # Get audio from input queue with timeout
                try:
                    audio_chunk = self.input_queue.get(timeout=QUEUE_TIMEOUT)
                    audio_buffer = np.append(audio_buffer, audio_chunk.flatten())
                except queue.Empty:
                    continue

                # Process when we have enough audio
                if len(audio_buffer) >= chunk_samples:
                    # Extract chunk
                    chunk = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples:]

                    # Process through STT -> TTS pipeline
                    result = self.process_audio_chunk(chunk)

                    if result:
                        audio_data, sr = result
                        # Add to output queue
                        self.output_queue.put((audio_data, sr))

            except Exception as e:
                print(f"Error in processing loop: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()

    def output_callback(self, outdata, frames, time_info, status):
        """Callback for audio output stream"""
        if status:
            print(f"Output callback status: {status}", file=sys.stderr)

        try:
            # Try to get audio from output queue
            if not self.output_queue.empty():
                audio_data, sr = self.output_queue.get_nowait()

                # Resample if needed
                if self.output_sample_rate and sr != self.output_sample_rate:
                    from scipy import signal

                    num_samples = int(len(audio_data) * self.output_sample_rate / sr)
                    audio_data = signal.resample(audio_data, num_samples)

                # Ensure mono
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # Fill output buffer
                if len(audio_data) >= frames:
                    outdata[:] = audio_data[:frames].reshape(-1, 1)
                    # Put remaining audio back in queue if any
                    if len(audio_data) > frames:
                        self.output_queue.put((audio_data[frames:], self.output_sample_rate or sr))
                else:
                    outdata[: len(audio_data)] = audio_data.reshape(-1, 1)
                    outdata[len(audio_data) :] = 0
            else:
                # No audio available, output silence
                outdata.fill(0)

        except Exception as e:
            print(f"Error in output callback: {e}", file=sys.stderr)
            outdata.fill(0)

    def start(self, input_device=None, output_device=None):
        """Start the voice cloning pipeline"""
        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()

        # Start audio streams
        print(f"\nStarting audio streams...")
        print(f"Input device: {input_device or 'default'}")
        print(f"Output device: {output_device or 'default (use virtual device)'}")

        with sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
        ):
            with sd.OutputStream(
                device=output_device,
                channels=1,
                samplerate=self.output_sample_rate or DEFAULT_OUTPUT_SAMPLE_RATE,
                callback=self.output_callback,
            ):
                print("\n🎤 Voice cloning active! Speak into your microphone.")
                print("Press Ctrl+C to stop.\n")

                try:
                    while self.running:
                        time.sleep(MAIN_LOOP_SLEEP)
                except KeyboardInterrupt:
                    print("\nStopping...")

        self.stop()

    def stop(self):
        """Stop the voice cloning pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()


def list_audio_devices():
    """List available audio devices"""
    print("\n=== Available Audio Devices ===\n")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"   Max input channels: {device['max_input_channels']}")
        print(f"   Max output channels: {device['max_output_channels']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Voice Cloning Virtual Audio Device (Local Qwen3-TTS)"
    )
    parser.add_argument("voice_sample", help="Path to audio file containing voice to clone")
    parser.add_argument(
        "--voice-text",
        help="Transcript of the voice sample (required unless --x-vector-only)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model name or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    # Detect available device
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.version, "hip") and torch.version.hip is not None:
        default_device = "cuda"  # ROCm uses 'cuda' as device name
    else:
        default_device = "cpu"

    parser.add_argument(
        "--device",
        default=default_device,
        help="Device for models (cuda/cpu). AMD GPUs use 'cuda' with ROCm.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        help="Input audio device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        help="Output audio device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices and exit"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Duration of audio chunks to process in seconds (default: 3.0)",
    )
    parser.add_argument("--no-vad", action="store_true", help="Disable Voice Activity Detection")
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use only speaker embedding (lower quality, no transcript needed)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Validate voice text requirement
    if not args.x_vector_only and not args.voice_text:
        parser.error("--voice-text is required unless --x-vector-only is specified")

    # Create and start voice cloner
    cloner = VoiceCloner(
        voice_sample_path=args.voice_sample,
        voice_sample_text=args.voice_text,
        whisper_model_size=args.whisper_model,
        tts_model_name=args.tts_model,
        device=args.device,
        vad_filter=not args.no_vad,
        chunk_duration=args.chunk_duration,
        x_vector_only_mode=args.x_vector_only,
    )

    cloner.start(input_device=args.input_device, output_device=args.output_device)


if __name__ == "__main__":
    main()
