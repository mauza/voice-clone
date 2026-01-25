#!/usr/bin/env python3
"""
Voice Cloning Virtual Audio Device
Routes microphone input through voice cloning (STT→TTS) for use in conferencing apps
Converts your speech to text, then synthesizes it in the target voice
Uses local models - no data sent to cloud
"""

import argparse
import logging
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel
from scipy import signal
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Audio processing constants
SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000

# Processing parameters
QUEUE_TIMEOUT = 0.1  # 100ms for STT→TTS
MAIN_LOOP_SLEEP = 0.01


# ============================================================================
# VOICE CLONING PIPELINE - STT→TTS
# ============================================================================


class VoiceCloner:
    """Voice cloning pipeline using STT→TTS"""

    def __init__(
        self,
        voice_sample_path,
        voice_sample_text=None,
        whisper_model_size="base",
        tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device="cuda",
        compute_type=None,
        vad_filter=True,
        vad_threshold=0.5,
        x_vector_only_mode=False,
        chunk_duration=3.0,
        max_utterance_duration=10.0,
        silence_duration=0.8,
    ):
        self.voice_sample_path = Path(voice_sample_path)
        self.voice_sample_text = voice_sample_text
        self.sample_rate = SAMPLE_RATE
        self.x_vector_only_mode = x_vector_only_mode
        self.chunk_duration = chunk_duration
        self.max_utterance_duration = max_utterance_duration
        self.silence_duration = silence_duration
        self.device = device
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold

        if not self.voice_sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")

        # Load voice sample
        voice_audio, voice_sr = sf.read(str(self.voice_sample_path))
        logger.info(f"Voice sample loaded: {self.voice_sample_path}")

        # Initialize Silero VAD for better voice activity detection
        if vad_filter:
            logger.info("Loading Silero VAD model...")
            self.vad_model = load_silero_vad()
            logger.info(f"VAD threshold: {vad_threshold}")
        else:
            self.vad_model = None

        # Initialize Whisper for STT
        logger.info(f"Loading Whisper model ({whisper_model_size})...")
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"
        logger.info(f"Using compute type: {compute_type}")
        self.whisper_model = WhisperModel(
            whisper_model_size, device=device, compute_type=compute_type
        )

        # Initialize Qwen3-TTS
        logger.info(f"Loading Qwen3-TTS model ({tts_model_name})...")
        # Use sdpa for both CPU and GPU - works without flash-attn package
        attn_impl = "sdpa"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        logger.info(f"Using attention: {attn_impl}, dtype: {dtype}")
        self.tts_model = Qwen3TTSModel.from_pretrained(
            tts_model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        # Create voice clone prompt
        logger.info("Creating voice clone prompt from sample...")
        self.voice_clone_prompt = self.tts_model.create_voice_clone_prompt(
            ref_audio=str(self.voice_sample_path),
            ref_text=self.voice_sample_text,
            x_vector_only_mode=self.x_vector_only_mode,
        )
        logger.info("Voice clone prompt ready!")

        self.output_sample_rate = None
        self.output_stream_sample_rate = None

        # Audio buffers
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.output_buffer = np.array([], dtype=np.float32)

        # Control flags
        self.running = False
        self.processing_thread = None

        # Performance tracking
        self.total_chunks = 0
        self.total_processing_time = 0

        # Recording
        self.recorded_audio = []
        self.recording_file = None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.input_queue.put(indata.copy())

    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio through the pipeline (VAD already applied)"""
        tmp_input_path = None
        try:
            start_time = time.time()

            # Log duration of audio being processed
            duration = len(audio_chunk) / self.sample_rate
            logger.info(f"Processing utterance: {duration:.2f}s")

            # Save audio chunk to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
                sf.write(tmp_input.name, audio_chunk, self.sample_rate)
                tmp_input_path = tmp_input.name

            # Speech-to-Text with Whisper (improved settings)
            segments, info = self.whisper_model.transcribe(
                tmp_input_path,
                language="en",
                beam_size=5,  # Better accuracy
                best_of=5,  # Sample multiple times
                temperature=0.0,  # Deterministic output
                vad_filter=False,  # We already did VAD with Silero
                condition_on_previous_text=True,  # Use context
            )

            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()

            if not text:
                logger.debug("No speech detected")
                return None

            logger.info(f"Transcribed: {text}")

            # Text-to-Speech with voice cloning
            logger.info(f"Starting TTS generation for: {text}")
            wavs, sr = self.tts_model.generate_voice_clone(
                text=text,
                language="Auto",
                voice_clone_prompt=self.voice_clone_prompt,
                max_new_tokens=2048,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
            )

            if self.output_sample_rate is None:
                self.output_sample_rate = sr
                logger.info(f"TTS output sample rate: {sr}Hz")

            if wavs and len(wavs) > 0:
                audio_data = wavs[0]
                logger.info(f"Generated TTS audio: {len(audio_data)} samples at {sr}Hz")

                # Record if enabled
                if self.recording_file:
                    self.recorded_audio.append((audio_data.copy(), sr))

                result = (audio_data, sr)
            else:
                logger.warning("TTS returned no audio data (empty or None)")
                result = None

            # Track performance
            processing_time = time.time() - start_time
            self.total_chunks += 1
            self.total_processing_time += processing_time

            # Log every 50 chunks
            if self.total_chunks % 50 == 0:
                avg_time = self.total_processing_time / self.total_chunks
                chunk_duration = len(audio_chunk) / self.sample_rate
                realtime_factor = chunk_duration / avg_time if avg_time > 0 else 0
                logger.info(
                    f"Performance: {avg_time * 1000:.1f}ms avg processing, "
                    f"{realtime_factor:.1f}x realtime"
                )

            return result

        except Exception as e:
            logger.exception(f"Error processing audio chunk: {e}")

        finally:
            if tmp_input_path and Path(tmp_input_path).exists():
                Path(tmp_input_path).unlink()

        return None

    def processing_loop(self):
        """Main processing loop - uses utterance-based detection"""
        audio_buffer = np.array([], dtype=np.float32)
        last_speech_end = None
        speech_started = False

        max_buffer_samples = int(self.max_utterance_duration * self.sample_rate)
        silence_samples = int(self.silence_duration * self.sample_rate)
        vad_check_interval = int(0.5 * self.sample_rate)  # Check VAD every 0.5s

        logger.info(f"Processing loop started (utterance-based, max {self.max_utterance_duration}s, {self.silence_duration}s silence cutoff)")

        while self.running:
            try:
                # Get audio from input queue
                try:
                    audio_chunk = self.input_queue.get(timeout=QUEUE_TIMEOUT)
                    audio_buffer = np.append(audio_buffer, audio_chunk.flatten())
                except queue.Empty:
                    # Check if we have a pending utterance that timed out
                    if len(audio_buffer) > 0 and speech_started:
                        if last_speech_end is not None:
                            silence_samples_accumulated = len(audio_buffer) - last_speech_end
                            if silence_samples_accumulated >= silence_samples:
                                # Process the utterance (speech ended)
                                utterance = audio_buffer[:last_speech_end]
                                audio_buffer = audio_buffer[last_speech_end:]
                                last_speech_end = None
                                speech_started = False

                                result = self.process_audio_chunk(utterance)
                                if result:
                                    audio_data, sr = result
                                    if sr != self.output_sample_rate:
                                        self.output_sample_rate = sr
                                    self.output_queue.put((audio_data, sr))
                    continue

                # Check buffer periodically with VAD (every 0.5s of audio or when buffer is large)
                if len(audio_buffer) >= vad_check_interval or len(audio_buffer) >= max_buffer_samples:
                    if self.vad_model is not None:
                        audio_tensor = torch.FloatTensor(audio_buffer)

                        speech_timestamps = get_speech_timestamps(
                            audio_tensor,
                            self.vad_model,
                            threshold=self.vad_threshold,
                            sampling_rate=self.sample_rate,
                            min_speech_duration_ms=100,
                            min_silence_duration_ms=int(self.silence_duration * 1000),
                            speech_pad_ms=100,
                        )

                        if speech_timestamps:
                            speech_started = True
                            # Find the last speech segment end
                            last_speech_end = speech_timestamps[-1]['end']

                            # Check if we have enough silence after the last speech
                            silence_after = len(audio_buffer) - last_speech_end

                            if silence_after >= silence_samples:
                                # Complete utterance detected - process it
                                utterance = audio_buffer[:last_speech_end]
                                audio_buffer = audio_buffer[last_speech_end:]
                                last_speech_end = None
                                speech_started = False

                                result = self.process_audio_chunk(utterance)
                                if result:
                                    audio_data, sr = result
                                    if sr != self.output_sample_rate:
                                        self.output_sample_rate = sr
                                    self.output_queue.put((audio_data, sr))
                        else:
                            # No speech detected - clear old buffer if no speech started
                            if not speech_started and len(audio_buffer) > max_buffer_samples:
                                audio_buffer = audio_buffer[-silence_samples:]  # Keep only recent silence
                    else:
                        # No VAD - fall back to fixed chunks
                        chunk_samples = int(self.chunk_duration * self.sample_rate)
                        if len(audio_buffer) >= chunk_samples:
                            chunk = audio_buffer[:chunk_samples]
                            audio_buffer = audio_buffer[chunk_samples:]

                            result = self.process_audio_chunk(chunk)
                            if result:
                                audio_data, sr = result
                                if sr != self.output_sample_rate:
                                    self.output_sample_rate = sr
                                self.output_queue.put((audio_data, sr))

                # Prevent buffer from growing too large
                if len(audio_buffer) > max_buffer_samples:
                    if speech_started and last_speech_end is not None:
                        # Force process the current utterance
                        utterance = audio_buffer[:last_speech_end]
                        audio_buffer = audio_buffer[last_speech_end:]
                        last_speech_end = None
                        speech_started = False

                        result = self.process_audio_chunk(utterance)
                        if result:
                            audio_data, sr = result
                            if sr != self.output_sample_rate:
                                self.output_sample_rate = sr
                            self.output_queue.put((audio_data, sr))
                    else:
                        # No speech or can't determine end - just process what we have
                        audio_buffer = audio_buffer[-silence_samples:]

            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")

    def output_callback(self, outdata, frames, time_info, status):
        """Callback for audio output stream"""
        if status:
            logger.warning(f"Output callback status: {status}")

        try:
            # Fill buffer from queue
            while not self.output_queue.empty() and len(self.output_buffer) < frames * 2:
                audio_data, sr = self.output_queue.get_nowait()

                # Resample TTS output (usually 24000Hz) to output stream sample rate
                if sr != self.output_stream_sample_rate:
                    num_samples = int(len(audio_data) * self.output_stream_sample_rate / sr)
                    audio_data = signal.resample(audio_data, num_samples)

                # Ensure mono and flatten
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                self.output_buffer = np.append(self.output_buffer, audio_data)

            # Fill output from buffer
            if len(self.output_buffer) >= frames:
                outdata[:] = self.output_buffer[:frames].reshape(-1, 1)
                self.output_buffer = self.output_buffer[frames:]
            else:
                # Not enough buffered - output what we have and pad with zeros
                if len(self.output_buffer) > 0:
                    outdata[: len(self.output_buffer)] = self.output_buffer.reshape(-1, 1)
                    outdata[len(self.output_buffer) :] = 0
                    self.output_buffer = np.array([], dtype=np.float32)
                else:
                    outdata.fill(0)

        except Exception as e:
            logger.error(f"Error in output callback: {e}")
            outdata.fill(0)

    def start(self, input_device=None, output_device=None, output_stream_sr=None):
        """Start the voice cloning pipeline"""
        self.running = True

        # Set output stream sample rate
        if output_stream_sr:
            self.output_stream_sample_rate = output_stream_sr
        elif output_device is not None:
            try:
                device_info = sd.query_devices(output_device, "output")
                self.output_stream_sample_rate = int(device_info["default_samplerate"])
                logger.info(f"Output device sample rate: {self.output_stream_sample_rate}Hz")
            except:
                self.output_stream_sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE
        else:
            self.output_stream_sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE

        # TTS output sample rate (for resampling)
        if self.output_sample_rate is None:
            self.output_sample_rate = self.output_stream_sample_rate

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()

        # Start audio streams
        logger.info("Starting audio streams...")
        logger.info(f"Input device: {input_device or 'default'}")
        logger.info(f"Output device: {output_device or 'default'}")
        logger.info(f"Sample rate: {self.sample_rate}Hz")
        logger.info(f"Output stream sample rate: {self.output_stream_sample_rate}Hz")
        logger.info(f"Chunk duration: {self.chunk_duration}s")

        # Use larger blocksize for more stable buffering
        blocksize = int(self.sample_rate * 0.1)  # 100ms blocks

        with sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=blocksize,
            callback=self.audio_callback,
        ):
            with sd.OutputStream(
                device=output_device,
                channels=1,
                samplerate=self.output_stream_sample_rate,
                blocksize=blocksize,
                callback=self.output_callback,
            ):
                logger.info("🎤 Voice cloning active! Speak into your microphone.")
                logger.info("Press Ctrl+C to stop.")

                try:
                    while self.running:
                        time.sleep(MAIN_LOOP_SLEEP)
                except KeyboardInterrupt:
                    logger.info("Stopping...")

        self.stop()

    def stop(self):
        """Stop the voice cloning pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

        # Save recording if enabled
        if self.recording_file and self.recorded_audio:
            logger.info(f"Saving recording to {self.recording_file}...")
            # Combine all audio chunks
            all_audio = []
            for audio, sr in self.recorded_audio:
                all_audio.append(audio)
            combined = np.concatenate(all_audio)
            sf.write(self.recording_file, combined, 24000)
            logger.info(f"Recording saved: {len(combined)} samples")


# ============================================================================
# UTILITIES
# ============================================================================


def find_virtual_sink():
    """Find the virtual sink for voice clone output. Returns (device_index, sample_rate) or (None, None)"""
    try:
        devices = sd.query_devices()
        voice_idx = None
        voice_sr = None
        pulse_idx = None

        for i, device in enumerate(devices):
            name = device["name"].lower()
            # Look for the specific "Voice" device (virtual sink in PipeWire)
            if name == "voice" and device["max_output_channels"] > 0:
                voice_idx = i
                voice_sr = int(device["default_samplerate"])
            # Find PulseAudio backend as fallback
            if device["name"] == "pulse":
                pulse_idx = i

        # Prioritize PulseAudio backend (works with default sink routing)
        if pulse_idx is not None:
            logger.info(
                f"Using PulseAudio backend (index {pulse_idx}) - outputting to default sink"
            )
            return pulse_idx, 48000
        elif voice_idx is not None:
            logger.info(f"Found virtual sink: Voice (index {voice_idx}, sr: {voice_sr}Hz)")
            return voice_idx, voice_sr
    except Exception as e:
        logger.warning(f"Could not find virtual sink: {e}")
    return None, None


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
        description="Voice Cloning Virtual Audio Device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voice cloning using STT→TTS pipeline (2-5s latency, 8-12GB VRAM)

EXAMPLES:
  # With voice transcript (recommended)
  %(prog)s voice.wav --voice-text "Your transcript"

  # With speaker embedding only (no transcript needed)
  %(prog)s voice.wav --x-vector-only

  # List audio devices
  %(prog)s voice.wav --list-devices
""",
    )

    parser.add_argument("voice_sample", help="Path to audio file containing voice to clone")

    # Common options
    parser.add_argument("--input-device", type=int, help="Input audio device index")
    parser.add_argument("--output-device", type=int, help="Output audio device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--chunk-duration", type=float, help="Audio chunk duration in seconds")

    # Detect available device
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.version, "hip") and torch.version.hip is not None:
        default_device = "cuda"
    else:
        default_device = "cpu"

    parser.add_argument("--device", default=default_device, help="Device for models (cuda/cpu)")

    # Voice cloning options
    parser.add_argument("--voice-text", help="Transcript of voice sample (recommended)")
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small for better accuracy)",
    )
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model name (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.35,
        help="VAD sensitivity (0.0-1.0, lower = more sensitive, default: 0.35)",
    )
    parser.add_argument(
        "--max-utterance",
        type=float,
        default=10.0,
        help="Maximum utterance duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.8,
        help="Silence duration to detect end of sentence in seconds (default: 0.8)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use only speaker embedding (no transcript needed)",
    )
    parser.add_argument(
        "--record-output",
        help="Record all TTS output to a WAV file (e.g., output.wav)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Auto-detect virtual sink for output if not specified
    output_device = args.output_device
    if output_device is None:
        virtual_sink, sink_sr = find_virtual_sink()
        if virtual_sink is not None:
            output_device = virtual_sink
            logger.info(f"Auto-detected virtual sink as output device")
        else:
            logger.warning("Virtual sink not found. Outputting to default device.")
            logger.info("Run './run.sh setup' to create virtual sink for use as microphone input.")
            sink_sr = None

    # Verify voice sample exists
    if not Path(args.voice_sample).exists():
        print(f"Error: Voice sample not found: {args.voice_sample}", file=sys.stderr)
        sys.exit(1)

    # Validate requirements
    if not args.x_vector_only and not args.voice_text:
        parser.error("--voice-text required (unless --x-vector-only)")

    # Create voice cloner
    cloner = VoiceCloner(
        voice_sample_path=args.voice_sample,
        voice_sample_text=args.voice_text,
        whisper_model_size=args.whisper_model,
        tts_model_name=args.tts_model,
        device=args.device,
        vad_filter=not args.no_vad,
        vad_threshold=args.vad_threshold,
        x_vector_only_mode=args.x_vector_only,
        chunk_duration=args.chunk_duration or 3.0,
        max_utterance_duration=args.max_utterance,
        silence_duration=args.silence_duration,
    )
    cloner.recording_file = args.record_output

    cloner.start(
        input_device=args.input_device, output_device=output_device, output_stream_sr=sink_sr
    )


if __name__ == "__main__":
    main()
