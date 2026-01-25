#!/usr/bin/env python3
"""
Voice Cloning Virtual Audio Device
Routes microphone input through voice cloning models for use in conferencing apps
Supports two modes: Fast (Voice Conversion) and Quality (STT→TTS)
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Audio processing constants
WHISPER_SAMPLE_RATE = 16000
VC_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000

# Processing parameters
QUEUE_TIMEOUT_FAST = 0.01  # 10ms for voice conversion
QUEUE_TIMEOUT_QUALITY = 0.1  # 100ms for STT→TTS
MAIN_LOOP_SLEEP = 0.01


# ============================================================================
# VOICE CONVERSION MODE - Fast, Low Latency
# ============================================================================

class VoiceConverterBase:
    """Base class for voice conversion models"""

    def __init__(self, voice_sample_path, device="cuda", sample_rate=16000):
        self.voice_sample_path = Path(voice_sample_path)
        self.device = device
        self.sample_rate = sample_rate

        if not self.voice_sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")

        # Load and prepare voice sample
        logger.info(f"Loading voice sample: {self.voice_sample_path}")
        self.reference_audio, ref_sr = sf.read(str(self.voice_sample_path))

        # Resample if needed
        if ref_sr != self.sample_rate:
            logger.info(f"Resampling voice sample from {ref_sr}Hz to {self.sample_rate}Hz")
            self.reference_audio = self._resample(self.reference_audio, ref_sr, self.sample_rate)

        # Convert to mono if stereo
        if len(self.reference_audio.shape) > 1:
            self.reference_audio = self.reference_audio.mean(axis=1)

        logger.info(f"Reference audio: {len(self.reference_audio)} samples at {self.sample_rate}Hz")

    def _resample(self, audio, orig_sr, target_sr):
        """Resample audio using torchaudio"""
        audio_tensor = torch.from_numpy(audio).float()
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        resampled = torchaudio.transforms.Resample(orig_sr, target_sr)(audio_tensor)
        return resampled.squeeze(0).numpy()

    def convert(self, audio_chunk):
        """Convert voice in audio chunk - to be implemented by subclasses"""
        raise NotImplementedError


class SimpleVoiceConverter(VoiceConverterBase):
    """Simple voice conversion using pitch and formant shifting"""

    def __init__(self, voice_sample_path, device="cuda", sample_rate=16000,
                 pitch_shift=0, formant_shift=1.0):
        super().__init__(voice_sample_path, device, sample_rate)
        self.pitch_shift = pitch_shift
        self.formant_shift = formant_shift
        self._analyze_reference()
        logger.info("Initialized SimpleVoiceConverter (pitch/formant shifting)")

    def _analyze_reference(self):
        """Analyze reference audio for pitch characteristics"""
        try:
            import librosa
            import numpy as np
            # Extract fundamental frequency using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                self.reference_audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            # Get median pitch of voiced segments
            self.ref_pitch = np.nanmedian(f0[voiced_flag]) if np.any(voiced_flag) else 0
            logger.info(f"Reference pitch: {self.ref_pitch:.2f} Hz")
        except ImportError:
            logger.warning("librosa not available, using basic pitch estimation")
            self.ref_pitch = 0

    def convert(self, audio_chunk):
        """Apply pitch and formant shifting"""
        try:
            import librosa

            # Apply pitch shifting (skip if pitch_shift is 0)
            if self.pitch_shift == 0:
                shifted = audio_chunk
            else:
                shifted = librosa.effects.pitch_shift(
                    audio_chunk,
                    sr=self.sample_rate,
                    n_steps=self.pitch_shift
                )

            # Apply formant shifting via time stretching
            if self.formant_shift != 1.0:
                shifted = librosa.effects.time_stretch(shifted, rate=self.formant_shift)

            return shifted

        except ImportError:
            logger.warning("librosa not available, voice conversion disabled")
            return audio_chunk


class RVCVoiceConverter(VoiceConverterBase):
    """RVC (Retrieval-based Voice Conversion) implementation"""

    def __init__(self, voice_sample_path, model_path=None, device="cuda", sample_rate=16000):
        super().__init__(voice_sample_path, device, sample_rate)

        try:
            from rvc_infer import RVC

            if model_path is None:
                raise ValueError("RVC model path required. Use --model-path argument.")

            if not Path(model_path).exists():
                raise FileNotFoundError(f"RVC model not found: {model_path}")

            logger.info(f"Loading RVC model from {model_path}")
            self.model = RVC(
                model_path=model_path,
                device=device,
                sample_rate=sample_rate
            )

            logger.info("Extracting speaker embedding from reference audio...")
            self.speaker_embedding = self.model.get_speaker_embedding(self.reference_audio)
            logger.info("RVC model ready!")

        except ImportError:
            raise ImportError(
                "RVC dependencies not installed. Install with:\n"
                "uv pip install rvc-python faiss-cpu praat-parselmouth"
            )

    def convert(self, audio_chunk):
        """Convert voice using RVC"""
        converted = self.model.infer(
            audio_chunk,
            speaker_embedding=self.speaker_embedding,
            f0_method="rmvpe",
            f0_up_key=0,
            index_rate=0.5,
        )
        return converted


# ============================================================================
# STT→TTS MODE - High Quality, Higher Latency
# ============================================================================

class STTTTSPipeline:
    """Original STT→TTS pipeline for maximum quality"""

    def __init__(
        self,
        voice_sample_path,
        voice_sample_text=None,
        whisper_model_size="base",
        tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device="cuda",
        compute_type=None,
        vad_filter=True,
        x_vector_only_mode=False,
    ):
        self.voice_sample_path = Path(voice_sample_path)
        self.voice_sample_text = voice_sample_text
        self.sample_rate = WHISPER_SAMPLE_RATE
        self.x_vector_only_mode = x_vector_only_mode

        if not self.voice_sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")

        # Load voice sample
        voice_audio, voice_sr = sf.read(str(self.voice_sample_path))
        logger.info(f"Voice sample loaded: {self.voice_sample_path}")

        # Initialize Whisper for STT
        logger.info(f"Loading Whisper model ({whisper_model_size})...")
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"
        logger.info(f"Using compute type: {compute_type}")
        self.whisper_model = WhisperModel(
            whisper_model_size, device=device, compute_type=compute_type
        )
        self.vad_filter = vad_filter

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

    def process(self, audio_chunk):
        """Process audio through STT→TTS pipeline"""
        tmp_input_path = None
        try:
            # Save audio chunk to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
                sf.write(tmp_input.name, audio_chunk, self.sample_rate)
                tmp_input_path = tmp_input.name

            # Speech-to-Text with Whisper
            segments, info = self.whisper_model.transcribe(
                tmp_input_path, vad_filter=self.vad_filter, language="en"
            )

            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()

            if not text:
                logger.debug("No speech detected")
                return None

            logger.info(f"Transcribed: {text}")

            # Text-to-Speech with voice cloning
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
                logger.debug(f"Generated TTS audio: {len(audio_data)} samples at {sr}Hz")
                return audio_data, sr

        except Exception as e:
            logger.exception(f"Error processing audio chunk: {e}")

        finally:
            if tmp_input_path and Path(tmp_input_path).exists():
                Path(tmp_input_path).unlink()

        return None


# ============================================================================
# UNIFIED VOICE CLONER
# ============================================================================

class VoiceCloner:
    """Unified voice cloner supporting both fast (VC) and quality (STT→TTS) modes"""

    def __init__(self, mode="fast", **kwargs):
        """
        Initialize voice cloner

        Args:
            mode: "fast" for voice conversion, "quality" for STT→TTS
            **kwargs: Mode-specific arguments
        """
        self.mode = mode
        self.chunk_duration = kwargs.get('chunk_duration', 0.5 if mode == "fast" else 3.0)
        self.sample_rate = VC_SAMPLE_RATE if mode == "fast" else WHISPER_SAMPLE_RATE

        # Initialize appropriate pipeline
        if mode == "fast":
            converter_type = kwargs.get('converter_type', 'simple')
            if converter_type == 'simple':
                self.pipeline = SimpleVoiceConverter(
                    kwargs['voice_sample_path'],
                    device=kwargs.get('device', 'cuda'),
                    sample_rate=self.sample_rate
                )
            elif converter_type == 'rvc':
                self.pipeline = RVCVoiceConverter(
                    kwargs['voice_sample_path'],
                    model_path=kwargs.get('model_path'),
                    device=kwargs.get('device', 'cuda'),
                    sample_rate=self.sample_rate
                )
            else:
                raise ValueError(f"Unknown converter type: {converter_type}")

        elif mode == "quality":
            self.pipeline = STTTTSPipeline(
                voice_sample_path=kwargs['voice_sample_path'],
                voice_sample_text=kwargs.get('voice_sample_text'),
                whisper_model_size=kwargs.get('whisper_model_size', 'base'),
                tts_model_name=kwargs.get('tts_model_name', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'),
                device=kwargs.get('device', 'cuda'),
                vad_filter=kwargs.get('vad_filter', True),
                x_vector_only_mode=kwargs.get('x_vector_only_mode', False),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Audio buffers
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.output_buffer = np.array([], dtype=np.float32)  # Persistent output buffer

        # Control flags
        self.running = False
        self.processing_thread = None

        # Performance tracking
        self.total_chunks = 0
        self.total_processing_time = 0

        # Output sample rate
        self.output_sample_rate = self.sample_rate

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.input_queue.put(indata.copy())

    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio through the pipeline"""
        try:
            start_time = time.time()

            if self.mode == "fast":
                result = self.pipeline.convert(audio_chunk)
                if result is not None:
                    result = (result, self.sample_rate)
            else:  # quality mode
                result = self.pipeline.process(audio_chunk)

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
                    f"Performance: {avg_time*1000:.1f}ms avg processing, "
                    f"{realtime_factor:.1f}x realtime"
                )

            return result

        except Exception as e:
            logger.exception(f"Error processing audio chunk: {e}")
            return None

    def processing_loop(self):
        """Main processing loop - runs in separate thread"""
        audio_buffer = np.array([], dtype=np.float32)
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        queue_timeout = QUEUE_TIMEOUT_FAST if self.mode == "fast" else QUEUE_TIMEOUT_QUALITY

        logger.info(f"Processing loop started ({self.mode} mode, {self.chunk_duration}s chunks)")

        while self.running:
            try:
                # Get audio from input queue
                try:
                    audio_chunk = self.input_queue.get(timeout=queue_timeout)
                    audio_buffer = np.append(audio_buffer, audio_chunk.flatten())
                except queue.Empty:
                    continue

                # Process when we have enough audio
                while len(audio_buffer) >= chunk_samples:
                    chunk = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples:]

                    result = self.process_audio_chunk(chunk)

                    if result:
                        audio_data, sr = result

                        # Update output sample rate if needed (for quality mode)
                        if self.mode == "quality" and sr != self.output_sample_rate:
                            self.output_sample_rate = sr

                        self.output_queue.put((audio_data, sr))

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

                # Resample if needed
                if sr != self.output_sample_rate:
                    num_samples = int(len(audio_data) * self.output_sample_rate / sr)
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
                    outdata[:len(self.output_buffer)] = self.output_buffer.reshape(-1, 1)
                    outdata[len(self.output_buffer):] = 0
                    self.output_buffer = np.array([], dtype=np.float32)
                else:
                    outdata.fill(0)

        except Exception as e:
            logger.error(f"Error in output callback: {e}")
            outdata.fill(0)

    def start(self, input_device=None, output_device=None):
        """Start the voice cloning pipeline"""
        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()

        # Start audio streams
        logger.info("Starting audio streams...")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Input device: {input_device or 'default'}")
        logger.info(f"Output device: {output_device or 'default'}")
        logger.info(f"Sample rate: {self.sample_rate}Hz")
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
                samplerate=self.output_sample_rate,
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


# ============================================================================
# UTILITIES
# ============================================================================

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
MODES:
  --mode fast     Voice conversion (100-300ms latency, 2-4GB VRAM)
  --mode quality  STT→TTS pipeline (2-5s latency, 8-12GB VRAM)

EXAMPLES:
  # Fast mode (no transcript needed)
  %(prog)s voice.wav --mode fast --converter simple

  # Fast mode with RVC (best quality)
  %(prog)s voice.wav --mode fast --converter rvc --model-path model.pth

  # Quality mode
  %(prog)s voice.wav --mode quality --voice-text "Your transcript"

  # List audio devices
  %(prog)s voice.wav --list-devices
"""
    )

    parser.add_argument("voice_sample", help="Path to audio file containing voice to clone")
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "quality"],
        help="Processing mode (default: fast)"
    )

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

    # Fast mode options
    fast_group = parser.add_argument_group('fast mode options')
    fast_group.add_argument(
        "--converter",
        default="simple",
        choices=["simple", "rvc"],
        help="Voice converter type (default: simple)"
    )
    fast_group.add_argument("--model-path", help="Path to RVC model file (required for RVC)")

    # Quality mode options
    quality_group = parser.add_argument_group('quality mode options')
    quality_group.add_argument("--voice-text", help="Transcript of voice sample")
    quality_group.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    quality_group.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model name (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
    )
    quality_group.add_argument("--no-vad", action="store_true", help="Disable VAD")
    quality_group.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use only speaker embedding (no transcript needed)"
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Verify voice sample exists
    if not Path(args.voice_sample).exists():
        print(f"Error: Voice sample not found: {args.voice_sample}", file=sys.stderr)
        sys.exit(1)

    # Validate mode-specific requirements
    if args.mode == "fast":
        if args.converter == "rvc" and not args.model_path:
            parser.error("--model-path required when using RVC converter")

        # Create voice cloner
        cloner = VoiceCloner(
            mode="fast",
            voice_sample_path=args.voice_sample,
            converter_type=args.converter,
            model_path=args.model_path,
            device=args.device,
            chunk_duration=args.chunk_duration or 0.5,
        )

    else:  # quality mode
        if not args.x_vector_only and not args.voice_text:
            parser.error("--voice-text required in quality mode (unless --x-vector-only)")

        # Create voice cloner
        cloner = VoiceCloner(
            mode="quality",
            voice_sample_path=args.voice_sample,
            voice_sample_text=args.voice_text,
            whisper_model_size=args.whisper_model,
            tts_model_name=args.tts_model,
            device=args.device,
            vad_filter=not args.no_vad,
            x_vector_only_mode=args.x_vector_only,
            chunk_duration=args.chunk_duration or 3.0,
        )

    # Start voice cloning
    cloner.start(input_device=args.input_device, output_device=args.output_device)


if __name__ == "__main__":
    main()
