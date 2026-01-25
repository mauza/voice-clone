# Troubleshooting Guide

## Installation Issues

### "Import could not be resolved" errors

This is normal before installation. Run:
```bash
uv sync
```

### GPU Issues

**Error: CUDA/GPU not available**

For NVIDIA:
```bash
# Check if CUDA is installed
nvidia-smi

# Install CUDA-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For AMD:
```bash
# Check if ROCm detects your GPU
rocm-smi
rocminfo

# Install ROCm-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Error: CUDA out of memory / ROCm out of memory**
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

### AMD GPU Specific Issues

**FlashAttention errors on AMD GPU**

FlashAttention 2 may not work on all AMD GPUs. If you see errors:
```bash
# The model will automatically fall back to standard attention
# This is slower but should work on all GPUs
```

**ROCm not detecting GPU**
```bash
# Check ROCm installation
rocm-smi
rocminfo | grep "Name"

# Reinstall ROCm if needed (Arch Linux)
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# Check if PyTorch sees the GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Wrong ROCm version**
```bash
# Check your ROCm version
rocminfo | grep "ROCm version"

# Install matching PyTorch (e.g., for ROCm 6.0)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Missing System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev pulseaudio ffmpeg
```

**Arch Linux:**
```bash
sudo pacman -S portaudio pipewire pipewire-pulse ffmpeg
```

## Virtual Device Issues

### Virtual device not appearing

**Check if created:**
```bash
pactl list sinks short | grep voice_clone
```

**Recreate device:**
```bash
./setup_virtual_device.sh
```

**Restart audio system:**
```bash
# PulseAudio
pulseaudio --kill && pulseaudio --start

# PipeWire
systemctl --user restart pipewire pipewire-pulse
```

### Virtual device not showing in conferencing apps

1. Restart the conferencing app after creating virtual device
2. Check app audio permissions
3. Try setting as default device temporarily:
   ```bash
   pactl set-default-source voice_clone_output.monitor
   ```

## Audio Issues

### No audio output

**List devices:**
```bash
uv run voice_clone.py --list-devices
```

**Test with specific devices:**
```bash
uv run voice_clone.py sample.wav --input-device 0 --output-device 10
```

**Check volume:**
```bash
# List all sinks with volume
pactl list sinks | grep -A 10 voice_clone
```

### Audio is choppy or distorted

1. **Increase chunk duration:**
   ```bash
   uv run voice_clone.py sample.wav --chunk-duration 4.0
   ```

2. **Use GPU for faster processing:**
   ```bash
   uv run voice_clone.py sample.wav --device cuda
   ```

3. **Close other audio applications**

4. **Check system resources:**
   ```bash
   htop  # or top
   ```

### Echo or feedback

- Don't monitor the virtual device output while speaking
- Mute your speakers in the conferencing app
- Use headphones

## Transcription Issues

### No speech detected

1. **Check microphone:**
   ```bash
   # Record test audio
   arecord -d 5 test.wav
   aplay test.wav
   ```

2. **Disable VAD if it's cutting off speech:**
   ```bash
   uv run voice_clone.py sample.wav --no-vad
   ```

3. **Speak louder or closer to mic**

4. **Check input levels:**
   ```bash
   alsamixer  # or pavucontrol
   ```

### Poor transcription accuracy

1. **Use better Whisper model:**
   ```bash
   uv run voice_clone.py sample.wav --model medium
   # or
   uv run voice_clone.py sample.wav --model large
   ```

2. **Reduce background noise**

3. **Speak more clearly**

4. **Check if language is set correctly** (currently hardcoded to English)

### Wrong language detected

The script is currently set to English. To modify for other languages, edit `voice_clone.py`:

```python
# Find this line:
segments, info = self.whisper_model.transcribe(
    tmp_input_path,
    vad_filter=self.vad_filter,
    language="en"  # Change to your language code
)
```

## TTS/Voice Cloning Issues

### Poor voice cloning quality

1. **Use better voice sample:**
   - 10-30 seconds duration
   - Clear, high-quality audio
   - No background noise
   - Natural speech patterns

2. **Convert to WAV format:**
   ```bash
   ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
   ```

### API timeout errors

1. **Increase timeout** (edit script)

2. **Check internet connection:**
   ```bash
   ping huggingface.co
   ```

3. **Try again later** (Hugging Face may be busy)

### Rate limiting

Qwen3-TTS on Hugging Face Spaces may have rate limits.

**Solutions:**
- Wait a few minutes between requests
- Use longer chunk durations to reduce API calls
- Consider duplicating the Space for private use (requires HF account)

## Performance Issues

### High latency (>5 seconds)

1. **Use GPU:**
   ```bash
   uv run voice_clone.py sample.wav --device cuda
   ```

2. **Smaller model:**
   ```bash
   uv run voice_clone.py sample.wav --model tiny
   ```

3. **Shorter chunks:**
   ```bash
   uv run voice_clone.py sample.wav --chunk-duration 2.0
   ```

4. **Check network speed:**
   ```bash
   speedtest-cli
   ```

### High CPU/Memory usage

1. **Use smaller Whisper model**
2. **Increase chunk duration** (fewer API calls)
3. **Close other applications**
4. **Check for memory leaks:**
   ```bash
   python voice_clone.py sample.wav &
   top -p $!
   ```

## Connection Issues

### Cannot connect to Qwen3-TTS

**Check internet:**
```bash
curl https://huggingface.co/spaces/Qwen/Qwen3-TTS
```

**Check Gradio client:**
```python
from gradio_client import Client
client = Client("Qwen/Qwen3-TTS")
client.view_api()
```

**Alternative: Use different TTS**

If Qwen3-TTS is unavailable, you can modify the script to use other TTS services.

## Debugging

### Enable verbose logging

Edit `voice_clone.py` and add at the top:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test components individually

**Test Whisper:**
```python
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cuda")
segments, info = model.transcribe("test.wav")
for segment in segments:
    print(segment.text)
```

**Test Gradio client:**
```python
from gradio_client import Client, handle_file
client = Client("Qwen/Qwen3-TTS")
result = client.predict(
    text="Hello world",
    voice_sample=handle_file("sample.wav"),
    api_name="/predict"
)
print(result)
```

**Test audio devices:**
```python
import sounddevice as sd
sd.query_devices()  # List devices
sd.rec(16000, samplerate=16000, channels=1)  # Test recording
```

### Common Error Messages

**"FileNotFoundError: Voice sample not found"**
- Check the path to your voice sample file
- Use absolute path instead of relative path

**"OSError: PortAudio library not found"**
- Install portaudio: `sudo apt-get install portaudio19-dev`

**"RuntimeError: CUDA error: out of memory"**
- Use smaller model or switch to CPU
- Close other GPU-intensive applications

**"Connection refused" or "Network error"**
- Check internet connection
- Check firewall settings
- Try again later (HF may be down)

**"Module not found" errors**
- Run `uv sync`
- Check Python version (3.8+ required)

## Getting Help

If you're still having issues:

1. **Check the README.md** for usage examples
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Your system info (OS, Python version)
   - Full error message
   - Steps to reproduce
   - What you've already tried

### System Info for Bug Reports

```bash
# Collect system info
echo "OS: $(uname -a)"
echo "Python: $(python3 --version)"
echo "Audio system: $(pactl info | grep 'Server Name' || echo 'PipeWire')"
echo "CUDA: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo 'Not available')"
pip list | grep -E "gradio|whisper|torch|sounddevice"
```

## Known Limitations

1. **Latency**: Inherent 2-5 second delay due to processing pipeline
2. **API Rate Limits**: Hugging Face Spaces may throttle requests
3. **Voice Quality**: Depends heavily on sample quality
4. **Language Support**: Currently optimized for English only
5. **Real-time**: Not truly real-time, uses chunked processing

## Advanced Troubleshooting

### Monitor the pipeline

Add debug prints to `voice_clone.py`:

```python
# In process_audio_chunk:
print(f"DEBUG: Processing chunk of {len(audio_chunk)} samples")
print(f"DEBUG: Transcribed: '{text}'")
print(f"DEBUG: TTS result: {result}")
print(f"DEBUG: Generated audio: {len(audio_data)} samples at {sr}Hz")
```

### Profile performance

```python
import time

# Wrap sections with timing:
start = time.time()
# ... code ...
print(f"Section took {time.time() - start:.2f}s")
```

### Check audio quality

Save intermediate audio for inspection:

```python
# After STT, before TTS:
sf.write('debug_input.wav', audio_chunk, self.sample_rate)

# After TTS:
sf.write('debug_output.wav', audio_data, sr)
```
