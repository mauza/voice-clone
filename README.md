# Voice Cloning Virtual Audio Device

Transform your voice in real-time using AI voice cloning for use in conferencing apps like Google Meet, Zoom, Discord, etc.

**100% Local - No data sent to cloud!**

## How It Works

```
Your Mic → Speech-to-Text (Whisper) → Text-to-Speech (Qwen3-TTS) → Virtual Device → Conferencing App
         [Local GPU/CPU]                [Local GPU/CPU]
```

1. Captures audio from your microphone
2. Transcribes speech to text using Whisper (runs locally)
3. Synthesizes speech with cloned voice using Qwen3-TTS (runs locally)
4. Outputs to a virtual audio device
5. Select the virtual device as your microphone in conferencing apps

**Privacy**: All processing happens on your machine. No audio or text is sent to any cloud service.

## Requirements

- Linux with PulseAudio or PipeWire
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- **GPU strongly recommended** (models are large and GPU is needed for real-time performance)
  - NVIDIA GPU with CUDA support, OR
  - AMD GPU with ROCm support (see AMD GPU setup below)
- At least 8GB VRAM for 1.7B model, 4GB for 0.6B model
- A sample audio file of the voice you want to clone (WAV, MP3, etc.)
- Transcript of the voice sample (or use --x-vector-only mode for lower quality)

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for other methods.

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev pulseaudio ffmpeg
```

**Arch Linux:**
```bash
sudo pacman -S portaudio pipewire pipewire-pulse ffmpeg
```

### 3. Install Python Dependencies

```bash
uv sync
```

This will install all dependencies including Qwen3-TTS from GitHub.

### 4. Install PyTorch with GPU Support (Recommended)

**For NVIDIA GPUs (CUDA):**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For AMD GPUs (ROCm):**

First, ensure ROCm is installed on your system. On Arch Linux:
```bash
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk
```

Then install PyTorch with ROCm:
```bash
# For ROCm 6.0+
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Or for ROCm 5.7
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

Check your ROCm version:
```bash
rocminfo | grep "ROCm version" || echo "ROCm not found"
```

**For CPU only (not recommended for real-time):**
```bash
# CPU version is included in uv sync
```

### 5. Verify GPU Detection

**For NVIDIA:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**For AMD:**
```bash
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

You should see `True` and your GPU name. If not, check your ROCm/CUDA installation.

### 6. Set Up Virtual Audio Device

```bash
chmod +x setup_virtual_device.sh
./setup_virtual_device.sh
```

This creates a virtual audio sink called "Voice Clone Virtual Output" that will appear as a microphone option in your conferencing apps.

## AMD GPU Notes

AMD GPUs work with this project via ROCm! Key things to know:

- **Use `--device cuda`** even with AMD GPUs (ROCm uses the same device API)
- **ROCm 6.0+ recommended** for best compatibility
- **Supported AMD GPUs**: Check [ROCm compatibility list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- **FlashAttention 2** may not work on all AMD GPUs - if you get errors, the models will fall back to standard attention (slower but works)
- Performance may vary compared to NVIDIA GPUs depending on your specific AMD GPU model

**Troubleshooting AMD GPU issues:**
```bash
# Check if ROCm detects your GPU
rocm-smi

# Check PyTorch ROCm build
python -c "import torch; print(torch.version.hip)"

# If you see errors, try without FlashAttention
# (models will automatically fall back, but you can force it by not installing flash-attn)
```

### Getting the Voice Sample Transcript

You need the transcript of your voice sample. You can:

1. **Manually type it out** (most accurate)
2. **Use the helper script to transcribe it**:
   ```bash
   uv run transcribe_sample.py voice_sample.wav
   ```
   
   This will output the transcript and the command to run voice_clone.py

## Usage

### Basic Usage

**Important:** You need to provide the transcript of your voice sample!

```bash
uv run voice_clone.py voice_sample.wav --voice-text "Okay. Yeah. I resent you. I love you."
```

This will:
- Use the default microphone as input
- Clone the voice from `voice_sample.wav` using the provided transcript
- Output to the default output device

**Alternative:** Use `--x-vector-only` mode (lower quality, no transcript needed):

```bash
uv run voice_clone.py voice_sample.wav --x-vector-only
```

### List Available Audio Devices

```bash
uv run voice_clone.py --list-devices
```

Find the device indices for your microphone and virtual device.

### Specify Devices

```bash
uv run voice_clone.py path/to/voice_sample.wav \
  --input-device 0 \
  --output-device 10
```

### Advanced Options

```bash
uv run voice_clone.py voice_sample.wav \
  --voice-text "Your voice sample transcript here" \
  --whisper-model medium \      # Whisper model: tiny, base, small, medium, large
  --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \  # Use smaller/faster model
  --device cuda \               # Use GPU for faster processing
  --chunk-duration 2.0 \        # Process audio in 2-second chunks
  --no-vad                      # Disable Voice Activity Detection
```

**Available TTS Models:**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (default, best quality, needs ~8GB VRAM)
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (faster, needs ~4GB VRAM)

### Full Example

```bash
# 1. List devices to find your microphone and virtual device indices
uv run voice_clone.py voice_sample.wav --voice-text "temp" --list-devices

# Example output:
# 0: Built-in Microphone
# ...
# 12: Voice Clone Virtual Output
# ...

# 2. Run voice cloning
uv run voice_clone.py voice_sample.wav \
  --voice-text "This is my voice sample transcript" \
  --input-device 0 \
  --output-device 12 \
  --whisper-model base \
  --device cuda

# 3. In Google Meet/Zoom:
#    - Go to Settings → Audio
#    - Select "Voice Clone Virtual Output" as your microphone
#    - Start talking!
```

## Tips for Best Results

### Voice Sample Quality

- **Duration**: 10-30 seconds is ideal
- **Quality**: Clear, high-quality audio without background noise
- **Content**: Natural speech with varied intonation
- **Format**: WAV or high-quality MP3

### Performance Tuning

**Fast (Lower Quality):**
```bash
uv run voice_clone.py sample.wav \
  --voice-text "..." \
  --whisper-model tiny \
  --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --chunk-duration 2.0
```

**Balanced (Recommended):**
```bash
uv run voice_clone.py sample.wav \
  --voice-text "..." \
  --whisper-model base \
  --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --chunk-duration 3.0 \
  --device cuda
```

**Best Quality (Slower):**
```bash
uv run voice_clone.py sample.wav \
  --voice-text "..." \
  --whisper-model medium \
  --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --chunk-duration 4.0 \
  --device cuda
```

### Latency Considerations

- Smaller chunks = lower latency
- Larger chunks = higher latency but better quality  
- **GPU is essential** for real-time performance (CPU will be too slow)
- 0.6B model is faster than 1.7B model
- Expected latency: 2-5 seconds with GPU

## Troubleshooting

### Virtual Device Not Appearing

```bash
# Reload PulseAudio/PipeWire
pulseaudio --kill
pulseaudio --start

# Or re-run setup
./setup_virtual_device.sh
```

### No Audio Output

1. Check device indices with `--list-devices`
2. Ensure virtual device is created: `pactl list sinks short`
3. Check volume levels in system audio settings

### High Latency

- Use a smaller `--chunk-duration` (e.g., 2.0)
- Use `--whisper-model tiny` or `--whisper-model base`
- Use `--tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base` (smaller/faster)
- **Ensure you're using `--device cuda` if you have a GPU** (CPU will be very slow)
- Close other GPU-intensive applications

### Poor Transcription Quality

- Use a better Whisper model: `--whisper-model medium` or `--whisper-model large`
- Use `--no-vad` if VAD is cutting off speech
- Improve microphone quality/reduce background noise
- Speak more clearly with pauses between phrases

### Poor Voice Cloning Quality

- Ensure voice sample is high quality with no background noise
- Provide accurate transcript with `--voice-text`
- Use longer voice sample (10-30 seconds recommended)
- Try the larger model: `--tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base`

### CUDA Out of Memory

```bash
# Use smaller models
uv run voice_clone.py sample.wav \
  --voice-text "..." \
  --whisper-model tiny \
  --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --device cuda

# Or use CPU (will be VERY slow - not recommended for real-time)
uv run voice_clone.py sample.wav \
  --voice-text "..." \
  --whisper-model base \
  --device cpu
```

## Architecture

### Components

1. **Audio Input**: Captures mic audio using sounddevice
2. **Speech-to-Text**: Faster-Whisper for local transcription
3. **Voice Cloning**: Qwen3-TTS via Gradio Client API
4. **Audio Output**: Routes to virtual device via sounddevice
5. **Virtual Device**: PulseAudio/PipeWire null sink

### Processing Pipeline

```python
# Input: 16kHz mono audio from microphone
# ↓
# STT: Faster-Whisper (local, GPU/CPU)
# ↓
# Text transcription
# ↓
# TTS: Qwen3-TTS (Hugging Face API, with voice sample)
# ↓
# Output: 24kHz audio to virtual device
```

## Privacy & Security

- **Speech-to-Text**: Runs 100% locally on your machine (Whisper)
- **Text-to-Speech**: Runs 100% locally on your machine (Qwen3-TTS)
- **Voice Sample**: Stays on your machine, never uploaded
- **Network**: **No internet connection required** after model download

**Privacy Guarantee**: All audio processing happens entirely on your machine. No data is sent to any cloud service.

## Limitations

1. **Latency**: 2-5 seconds delay due to chunking and processing
2. **Hardware**: Requires powerful GPU (8GB+ VRAM recommended)
3. **Real-time**: Not truly real-time due to processing pipeline
4. **Voice Quality**: Depends on sample quality and transcript accuracy
5. **Languages**: Currently optimized for English (model supports 10 languages)
6. **First Run**: Model download can take time (~3-7GB depending on model)

## License

This project is for educational and experimental purposes. Please respect voice cloning ethics and laws.

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Local TTS option (no API calls)
- [ ] Better audio resampling
- [ ] GUI for easier setup
- [ ] Multiple voice profiles
- [ ] Real-time voice activity detection improvements
- [ ] Reduced latency optimizations

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Qwen team at Alibaba Cloud for open-source TTS with voice cloning
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
