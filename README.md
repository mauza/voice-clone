# Voice Cloning Virtual Audio Device

Transform your voice in real-time using AI voice cloning for use in conferencing apps like Google Meet, Zoom, Discord, etc.

**100% Local - No data sent to cloud!**

## Two Modes

### ⚡ Fast Mode (Voice Conversion)
- **Latency:** 100-300ms (real-time capable)
- **VRAM:** 2-4GB
- **No transcript needed**
- **Best for:** Video calls, gaming, live streaming

### 🎨 Quality Mode (STT→TTS)
- **Latency:** 2-5 seconds
- **VRAM:** 8-12GB
- **Requires transcript**
- **Best for:** Pre-recorded content, maximum quality

```
Fast:    Mic → [Voice Conversion] → Output           (100-300ms)
Quality: Mic → [Whisper STT] → [Qwen3 TTS] → Output (2-5s)
```

## Quick Start

```bash
# One-time setup (installs everything)
chmod +x run.sh
./run.sh setup

# Start voice cloning in fast mode
./run.sh start voice_sample.wav

# Or quality mode (with transcript)
./run.sh quality voice_sample.wav "Your transcript here"

# Check your hardware
./run.sh benchmark

# List audio devices
./run.sh list
```

## Installation

### Automatic Setup (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd voice-clone

# Run setup (installs everything)
chmod +x run.sh
./run.sh setup
```

This will:
- Install uv package manager (if needed)
- Install all Python dependencies
- Create virtual audio device
- Check for GPU support

### Manual Setup

If you prefer manual installation:

**1. System Dependencies**

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev pulseaudio ffmpeg
```

Arch Linux:
```bash
sudo pacman -S portaudio pipewire pipewire-pulse ffmpeg
```

**2. Python Dependencies**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install PyTorch with GPU support
# NVIDIA:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# AMD:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**3. Virtual Audio Device**

```bash
./run.sh setup  # Creates virtual device
```

### Optional: RVC for Best Quality

For highest quality voice conversion, install RVC after the main setup:

```bash
# After running ./run.sh setup, install RVC separately:
uv pip install rvc-python faiss-cpu praat-parselmouth

# Note: RVC dependencies may have version conflicts on some systems
# If installation fails, you can still use the simple converter
```

Then train an RVC model (see Training RVC Model section below).

## Usage

### Using run.sh (Recommended)

**Fast mode (default):**
```bash
# Simple converter (no training needed)
./run.sh start voice_sample.wav

# With RVC (best quality, requires trained model)
./run.sh start voice_sample.wav --converter rvc --model-path model.pth

# Adjust latency
./run.sh start voice_sample.wav --chunk-duration 0.3
```

**Quality mode:**
```bash
# With transcript
./run.sh quality voice_sample.wav "Your transcript here"

# With larger models
./run.sh quality voice_sample.wav "..." --whisper-model medium
```

**Utilities:**
```bash
# List audio devices
./run.sh list

# Check hardware
./run.sh benchmark

# Remove virtual device
./run.sh remove
```

### Direct Usage (Advanced)

You can also call voice_clone.py directly:

**Fast mode:**
```bash
uv run voice_clone.py voice_sample.wav --mode fast --converter simple

# With specific devices
uv run voice_clone.py voice.wav --mode fast --input-device 0 --output-device 12

# Adjust latency
uv run voice_clone.py voice.wav --mode fast --chunk-duration 0.3
```

**Quality mode:**
```bash
# Basic
uv run voice_clone.py voice_sample.wav --mode quality --voice-text "Transcript"

# With larger models
uv run voice_clone.py voice_sample.wav \
  --mode quality \
  --voice-text "..." \
  --whisper-model medium \
  --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base

# Without transcript
uv run voice_clone.py voice_sample.wav --mode quality --x-vector-only
```

## Training RVC Model (Optional, Best Quality)

For best voice conversion quality, train an RVC model:

### 1. Install RVC-WebUI

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI
pip install -r requirements.txt
```

### 2. Prepare Training Data

- Collect 5-10 minutes of clean voice recordings
- WAV format, 16kHz or 48kHz sample rate
- No background noise
- Natural speech with varied intonation

### 3. Train

```bash
# Start WebUI
python infer-web.py

# In browser (usually http://localhost:7865):
# 1. Go to "Training" tab
# 2. Upload audio files
# 3. Set experiment name (e.g., "my_voice")
# 4. Click: Process data → Extract features → Train
# 5. Wait 10-30 minutes on GPU
# 6. Model saved in logs/my_voice/
```

### 4. Use Your Model

```bash
uv run voice_clone.py voice_sample.wav \
  --mode fast \
  --converter rvc \
  --model-path ~/RVC-WebUI/logs/my_voice/model.pth
```

## Performance Comparison

| Feature | Fast Mode | Quality Mode |
|---------|-----------|--------------|
| **Latency** | 100-300ms | 2-5s |
| **VRAM** | 2-4GB | 8-12GB |
| **Transcript** | Not needed | Required |
| **Quality** | Good-Excellent | Excellent |
| **Prosody** | ✅ Preserved | ❌ Lost |
| **Real-time** | ✅ Yes | ❌ No |
| **CPU Viable** | ⚠️ Marginal | ❌ No |

### Fast Mode Converters

| Converter | Quality | Latency | VRAM | Training |
|-----------|---------|---------|------|----------|
| **Simple** | 6-7/10 | 120-180ms | 0.5GB | ❌ None |
| **RVC** | 9-9.5/10 | 150-200ms | 2.8GB | ✅ Required |

## Architecture

### Fast Mode (Voice Conversion)

```
Input Mic (16kHz)
     ↓
Voice Converter
     ├─ Simple: Pitch/formant shifting
     └─ RVC: Neural voice conversion
     ↓
Output (16kHz)
```

**Key characteristics:**
- Direct audio-to-audio transformation
- Preserves natural timing and prosody
- Single model pass
- Small chunks (0.3-0.7s)

### Quality Mode (STT→TTS)

```
Input Mic (16kHz)
     ↓
Whisper STT (local)
     ↓
Text transcription
     ↓
Qwen3-TTS (local, with voice cloning)
     ↓
Output (24kHz)
```

**Key characteristics:**
- Two-stage pipeline
- Text intermediate representation
- Larger chunks (3s)
- Higher quality, higher latency

## Use Cases

### Use Fast Mode For:
- ✅ Video conferencing (Google Meet, Zoom, Teams)
- ✅ Gaming voice chat (Discord, TeamSpeak)
- ✅ Live streaming (Twitch, YouTube Live)
- ✅ VoIP calls
- ✅ Real-time interaction
- ✅ Limited VRAM (<8GB)

### Use Quality Mode For:
- ✅ Video dubbing
- ✅ Podcast editing
- ✅ Audiobook creation
- ✅ Pre-recorded content
- ✅ Maximum quality needed
- ✅ When latency doesn't matter

## Tips for Best Results

### Voice Sample Quality
- **Duration:** 10-30 seconds (more for RVC training)
- **Quality:** Clean, high-quality recording
- **Environment:** Quiet, no background noise
- **Content:** Natural speech with varied intonation
- **Format:** WAV or high-quality MP3

### Latency Optimization
1. Use fast mode with `--chunk-duration 0.3`
2. Ensure GPU is being used (`--device cuda`)
3. Close other GPU applications
4. Use simple converter if RVC model unavailable

### Quality Optimization
1. Train an RVC model with 5-10 minutes of audio
2. Use quality mode for pre-recorded content
3. Provide accurate transcript for quality mode
4. Use larger models (medium Whisper, 1.7B TTS)

## Troubleshooting

### High Latency

```bash
# Check if GPU is being used
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reduce chunk size
./run.sh start voice.wav --chunk-duration 0.2

# Ensure no other apps using GPU
nvidia-smi  # or rocm-smi for AMD
```

### Poor Voice Quality

**Fast mode:** Simple converter has limited quality. Train an RVC model:
```bash
# See "Training RVC Model" section above
```

**Quality mode:** Use larger models:
```bash
./run.sh quality voice.wav "..." --whisper-model medium
```

### Out of Memory

```bash
# Use fast mode with simple converter
./run.sh start voice.wav --converter simple

# Or use smaller quality mode models
./run.sh quality voice.wav "..." --whisper-model tiny --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

### Audio Artifacts

```bash
# Increase chunk duration
./run.sh start voice.wav --chunk-duration 0.5

# Use better voice sample (longer, cleaner)
```

### Virtual Device Not Appearing

```bash
# Reload audio system
pulseaudio --kill && pulseaudio --start

# Or re-run setup
./run.sh setup

# Verify it exists
pactl list sinks short | grep Voice
```

## Hardware Requirements

### Minimum (Fast Mode, Simple Converter)
- **GPU:** 2GB VRAM (or CPU)
- **RAM:** 4GB
- **Python:** 3.11 or 3.12 (3.13+ not yet fully supported)
- **Performance:** 120-400ms latency

### Recommended (Fast Mode, RVC)
- **GPU:** 4GB+ VRAM (e.g., GTX 1650)
- **RAM:** 8GB
- **Python:** 3.11 or 3.12
- **Performance:** 150-200ms latency

### High-End (Quality Mode)
- **GPU:** 10GB+ VRAM (e.g., RTX 3080)
- **RAM:** 16GB
- **Python:** 3.11 or 3.12
- **Performance:** 2.5-3.5s latency

### Check Your Hardware

```bash
./run.sh benchmark
```

This will:
- Detect your GPU
- Estimate performance for both modes
- Recommend which mode to use

## Command Reference

### run.sh Commands

```bash
./run.sh setup                      # One-time setup
./run.sh start <voice_sample>       # Start fast mode
./run.sh quality <voice> <text>     # Start quality mode
./run.sh list                       # List audio devices
./run.sh benchmark                  # Check hardware
./run.sh remove                     # Remove virtual device
./run.sh help                       # Show help
```

### Fast Mode Examples
```bash
# Basic
./run.sh start voice.wav

# With RVC
./run.sh start voice.wav --converter rvc --model-path model.pth

# Low latency
./run.sh start voice.wav --chunk-duration 0.2

# High quality
./run.sh start voice.wav --chunk-duration 0.5
```

### Quality Mode Examples
```bash
# Basic
./run.sh quality voice.wav "Your transcript"

# Best quality (larger models)
./run.sh quality voice.wav "..." --whisper-model medium
```

### Direct voice_clone.py Usage
```bash
# Fast mode
uv run voice_clone.py voice.wav --mode fast [options]

# Quality mode
uv run voice_clone.py voice.wav --mode quality --voice-text "..." [options]

# Common options
--list-devices              # List audio devices
--input-device N            # Microphone device index
--output-device N           # Output device index
--device cuda/cpu           # Use GPU or CPU
--chunk-duration N          # Chunk size in seconds
```

## All Command-Line Options

```
voice_clone.py voice_sample.wav [options]

Required:
  voice_sample              Path to voice sample audio file

Mode Selection:
  --mode {fast,quality}     Processing mode (default: fast)

Common Options:
  --input-device N          Input audio device index
  --output-device N         Output audio device index
  --device {cuda,cpu}       Device for models (auto-detected)
  --chunk-duration N        Audio chunk duration in seconds
  --list-devices            List available audio devices

Fast Mode Options:
  --converter {simple,rvc}  Voice converter type (default: simple)
  --model-path PATH         RVC model file (required for RVC)

Quality Mode Options:
  --voice-text "..."        Transcript of voice sample
  --whisper-model SIZE      Whisper model: tiny, base, small, medium, large
  --tts-model MODEL         TTS model name (default: Qwen/...-1.7B-Base)
  --no-vad                  Disable Voice Activity Detection
  --x-vector-only           Use only speaker embedding (no transcript)
```

## Privacy & Security

- **Speech-to-Text:** Runs 100% locally (Whisper)
- **Text-to-Speech:** Runs 100% locally (Qwen3-TTS)
- **Voice Conversion:** Runs 100% locally
- **Voice Sample:** Never uploaded, stays on your machine
- **Network:** No internet required after model download

**Privacy Guarantee:** All audio processing happens entirely on your machine. No data sent to any cloud service.

## Limitations

1. **Latency:** Fast mode has 100-300ms delay, quality mode has 2-5s
2. **Hardware:** GPU strongly recommended (CPU is very slow)
3. **Voice Quality:** Depends on sample quality and mode/converter
4. **Languages:** Optimized for English (models support multiple languages)
5. **First Run:** Model download can take time (~3-7GB depending on mode)

## License

This project is for educational and experimental purposes. Please respect voice cloning ethics and laws.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - High-quality TTS with voice cloning
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Efficient speech recognition
- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - Voice conversion
- [librosa](https://github.com/librosa/librosa) - Audio processing

## Contributing

Contributions welcome! Areas for improvement:

- Additional voice conversion models
- Better audio preprocessing
- GUI for easier setup
- Real-time quality monitoring
- Reduced latency optimizations
- Multi-language support improvements
