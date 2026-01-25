# Voice Cloning Virtual Audio Device

Transform your voice in real-time using AI voice cloning for use in conferencing apps like Google Meet, Zoom, Discord, etc.

**100% Local - No data sent to cloud!**

## How It Works

The voice cloner outputs to a virtual audio sink, which creates a monitor source that you can select as your microphone in meeting apps:

```
Your Physical Mic
       ↓
Voice Cloner (STT → TTS)
       ↓
Virtual Sink (voice_clone_output)
       ↓
Virtual Monitor Source ← Use this as your mic in Google Meet/Zoom
```

## Voice Cloning (STT→TTS Pipeline)

- **Latency:** 1-3 seconds (utterance-based processing)
- **VRAM:** 8-12GB
- **Transcript:** Recommended (optional with `--x-vector-only`)
- **Best for:** High-quality voice transformation, voice dubbing, content creation
- **Processing:** Waits for complete sentences/thoughts (detected by pauses) before converting

```
Input Mic (16kHz)
     ↓
Silero VAD (utterance detection)
     ↓
Whisper STT (local)
     ↓
Text transcription
     ↓
Qwen3-TTS (local, with voice cloning)
     ↓
Virtual Sink (24kHz)
     ↓
Monitor Source ← Select this in Meet/Zoom
```

## Quick Start

```bash
# One-time setup (installs everything and creates virtual device)
chmod +x run.sh
./run.sh setup

# Start voice cloning with transcript (recommended)
./run.sh start voice_sample.wav "Your transcript here"

# Or start without transcript (uses speaker embedding only)
./run.sh start voice_sample.wav

# In Google Meet/Zoom, select "Voice Clone Virtual Output" as your microphone

# Check your hardware
./run.sh benchmark

# List audio devices to find the virtual device
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

## Usage

### Using run.sh (Recommended)

```bash
# With transcript (recommended)
./run.sh start voice_sample.wav "Your transcript here"

# Without transcript (uses speaker embedding)
./run.sh start voice_sample.wav

# With larger Whisper model
./run.sh start voice_sample.wav "Your transcript" --whisper-model medium

# List audio devices
./run.sh list

# Check hardware
./run.sh benchmark

# Remove virtual device
./run.sh remove
```

### Quality & Performance Tuning

The default configuration uses **Whisper "small"** model and **Silero VAD** for improved accuracy. You can adjust these settings:

**Whisper Model Selection:**
- `--whisper-model tiny`: Fastest, lowest quality (~32MB VRAM)
- `--whisper-model base`: Fast, decent quality (~74MB VRAM)
- `--whisper-model small`: **Default** - Good balance (~244MB VRAM)
- `--whisper-model medium`: Better accuracy (~769MB VRAM)
- `--whisper-model large`: Best accuracy (~1.5GB VRAM)

**Utterance Detection Settings:**
The system uses **utterance-based processing** - it waits for you to finish a complete sentence/thought before processing.

- `--silence-duration 0.5`: Process faster (shorter pause = end of sentence)
- `--silence-duration 0.8`: **Default** - Natural speech pausing
- `--silence-duration 1.2`: Wait longer (good for thoughtful speech)
- `--max-utterance 10.0`: **Default** - Maximum 10 seconds per utterance
- `--max-utterance 15.0`: Allow longer continuous speech

**VAD (Voice Activity Detection) Settings:**
- `--vad-threshold 0.2`: Very sensitive (picks up quieter speech, may include noise)
- `--vad-threshold 0.35`: **Default** - Balanced for continuous speech
- `--vad-threshold 0.5`: Less sensitive (only clear speech)
- `--no-vad`: Disable utterance detection (falls back to fixed 3s chunks)

**Example - Maximum quality:**
```bash
./run.sh start voice.wav "transcript" --whisper-model large --vad-threshold 0.3
```

**Example - Maximum speed:**
```bash
./run.sh start voice.wav "transcript" --whisper-model tiny --vad-threshold 0.5
```

**Example - For faster responses (shorter pauses):**
```bash
./run.sh start voice.wav "transcript" --silence-duration 0.5
```

**Example - For very fast/staccato speech:**
```bash
./run.sh start voice.wav "transcript" --vad-threshold 0.25 --silence-duration 0.6
```

**Example - For long-form continuous speech:**
```bash
./run.sh start voice.wav "transcript" --max-utterance 20.0 --silence-duration 1.0
```

### Direct Usage (Advanced)

You can also call voice_clone.py directly:

```bash
# With transcript
uv run voice_clone.py voice_sample.wav --voice-text "Transcript"

# With speaker embedding only (no transcript)
uv run voice_clone.py voice_sample.wav --x-vector-only

# With specific devices
uv run voice_clone.py voice.wav --voice-text "..." --input-device 0 --output-device 12

# With larger models
uv run voice_clone.py voice_sample.wav \
  --voice-text "..." \
  --whisper-model medium \
  --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --chunk-duration 2.0
```

## Architecture

```
Input Mic (16kHz)
     ↓
Silero VAD (voice activity detection)
     ↓
Whisper STT (local faster-whisper)
     ↓
Text transcription
     ↓
Qwen3-TTS (local, with voice cloning capability)
     ↓
Output (24kHz)
```

**How Utterance Detection Works:**

1. Audio continuously buffers from your microphone
2. Silero VAD detects when you start speaking
3. System waits for you to pause (default: 0.8s of silence)
4. Complete utterance is processed through Whisper → TTS
5. Your cloned voice plays while you speak the next sentence

This prevents mid-sentence cuts and provides more natural, complete responses.

**Key characteristics:**
- Two-stage STT→TTS pipeline
- Text intermediate representation
- Larger chunks (3s default, configurable with `--chunk-duration`)
- High quality voice synthesis in target voice
- Higher latency compared to direct voice conversion

## Use Cases

- ✅ Video dubbing
- ✅ Podcast editing
- ✅ Audiobook creation
- ✅ Pre-recorded content
- ✅ Voiceover work
- ✅ Content creation where quality > latency
- ✅ When you need maximum voice transformation quality

## Tips for Best Results

### Voice Sample Quality
- **Duration:** 10-30 seconds
- **Quality:** Clean, high-quality recording
- **Environment:** Quiet, no background noise
- **Content:** Natural speech with varied intonation
- **Format:** WAV or high-quality MP3

### Transcript Quality
- **Accuracy:** Provide accurate transcript of the voice sample
- **Best results:** When transcript matches the audio exactly
- **Alternative:** Use `--x-vector-only` if you don't have a transcript

### Latency vs Quality
- **Lower latency:** Use smaller `--chunk-duration` (1-2s)
- **Higher quality:** Use larger `--chunk-duration` (3-5s)

### Model Selection
- **Faster:** `--whisper-model tiny` or `base`
- **Better recognition:** `--whisper-model small` or `medium`
- **Better synthesis:** Larger TTS model names

## Troubleshooting

### High Latency

```bash
# Check if GPU is being used
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reduce chunk size
./run.sh start voice.wav "transcript" --chunk-duration 2.0

# Ensure no other apps using GPU
nvidia-smi  # or rocm-smi for AMD
```

### Poor Recognition Accuracy

```bash
# Use larger Whisper model
./run.sh start voice.wav "transcript" --whisper-model medium

# Disable VAD if it's cutting speech
./run.sh start voice.wav "transcript" --no-vad
```

### Poor Voice Output Quality

```bash
# Ensure transcript is accurate
# Use better voice sample (longer, cleaner)

# Check TTS model is loaded correctly
# Output should show "Voice clone prompt ready!"
```

### Out of Memory

```bash
# Use smaller models
uv run voice_clone.py voice.wav --voice-text "..." \
  --whisper-model tiny
```

### Audio Artifacts

```bash
# Increase chunk duration
./run.sh start voice.wav "transcript" --chunk-duration 4.0

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
pactl list sources short | grep Voice
```

### Using with Google Meet / Zoom / Discord

After running `./run.sh start voice_sample.wav "transcript"`:

1. The voice cloner outputs to the virtual sink `voice_clone_output`
2. This automatically creates a monitor source (e.g., `Voice Clone Virtual Output.monitor`)
3. In your meeting app, select the **monitor source** as your microphone input

**To find the correct device name:**
```bash
./run.sh list | grep -i "voice clone"
```

Look for the device with "monitor" in the name - this is the one to select as your microphone.

## Hardware Requirements

### Recommended
- **GPU:** 10GB+ VRAM (e.g., RTX 3080)
- **RAM:** 16GB
- **Python:** 3.11 or 3.12
- **Performance:** 2.5-3.5s latency

### Minimum
- **GPU:** 6GB VRAM (8GB recommended)
- **RAM:** 12GB
- **Python:** 3.11 or 3.12
- **Performance:** 3-5s latency (or much slower on CPU)

### Check Your Hardware

```bash
./run.sh benchmark
```

## Command Reference

### run.sh Commands

```bash
./run.sh setup                      # One-time setup
./run.sh start <voice_sample> [text] # Start voice cloning
./run.sh list                       # List audio devices
./run.sh benchmark                  # Check hardware
./run.sh remove                     # Remove virtual device
./run.sh help                       # Show help
```

### Examples

```bash
# Basic with transcript
./run.sh start voice.wav "Your transcript here"

# Start with speaker embedding (no transcript)
./run.sh start voice.wav

# With larger models
./run.sh start voice.wav "..." --whisper-model medium

# Adjust chunk duration
./run.sh start voice.wav "..." --chunk-duration 2.0
```

### Direct voice_clone.py Usage

```bash
uv run voice_clone.py voice.wav --voice-text "..." [options]

# Common options
--list-devices              # List audio devices
--input-device N            # Microphone device index
--output-device N           # Output device index
--device cuda/cpu           # Use GPU or CPU
--chunk-duration N          # Chunk size in seconds (default: 3.0)
--voice-text "..."          Transcript of voice sample
--x-vector-only             Use only speaker embedding (no transcript)
--whisper-model SIZE        Whisper model: tiny, base, small, medium, large
--tts-model MODEL           TTS model name
--no-vad                    Disable Voice Activity Detection
```

## All Command-Line Options

```
voice_clone.py voice_sample.wav [options]

Required:
  voice_sample              Path to voice sample audio file

Options:
  --voice-text "..."        Transcript of voice sample (recommended)
  --x-vector-only           Use only speaker embedding (no transcript needed)
  --whisper-model SIZE      Whisper model: tiny, base, small, medium, large (default: base)
  --tts-model MODEL         TTS model name (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
  --no-vad                  Disable Voice Activity Detection
  --input-device N          Input audio device index
  --output-device N         Output audio device index
  --device {cuda,cpu}       Device for models (auto-detected)
  --chunk-duration N        Audio chunk duration in seconds (default: 3.0)
  --list-devices            List available audio devices
```

## Privacy & Security

- **Speech-to-Text:** Runs 100% locally (Whisper)
- **Text-to-Speech:** Runs 100% locally (Qwen3-TTS)
- **Voice Sample:** Never uploaded, stays on your machine
- **Network:** No internet required after model download

**Privacy Guarantee:** All audio processing happens entirely on your machine. No data sent to any cloud service.

## Limitations

1. **Latency:** 2-5 seconds due to STT→TTS pipeline
2. **Hardware:** GPU strongly recommended (CPU is very slow)
3. **Voice Quality:** Depends on sample quality and transcript accuracy
4. **Languages:** Optimized for English (models support multiple languages)
5. **First Run:** Model download can take time (~3-5GB)
6. **Real-time:** Not suitable for real-time conversations due to latency

## License

This project is for educational and experimental purposes. Please respect voice cloning ethics and laws.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - High-quality TTS with voice cloning
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Efficient speech recognition

## Contributing

Contributions welcome! Areas for improvement:

- Better audio preprocessing
- GUI for easier setup
- Real-time quality monitoring
- Reduced latency optimizations
- Multi-language support improvements
