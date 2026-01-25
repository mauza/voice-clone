#!/bin/bash
# Quick start script for Voice Cloning Virtual Audio Device

set -e

echo "=== Voice Cloning Virtual Audio Device - Quick Start ==="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed!"
    echo ""
    echo "Install uv with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Install dependencies
echo "Step 1/3: Installing Python dependencies with uv..."
echo "This may take a few minutes (downloading models from GitHub)..."
uv sync

echo ""
echo "Step 2/3: Setting up virtual audio device..."
chmod +x setup_virtual_device.sh
./setup_virtual_device.sh

echo ""
echo "Step 3/3: Setup complete!"
echo ""
echo "=== GPU Support (Recommended) ==="
echo ""
echo "For NVIDIA GPUs (CUDA):"
echo "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "For AMD GPUs (ROCm):"
echo "  # First ensure ROCm is installed: sudo pacman -S rocm-hip-sdk rocm-opencl-sdk"
echo "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0"
echo ""
echo "Verify GPU detection:"
echo "  python -c \"import torch; print('GPU available:', torch.cuda.is_available())\""
echo ""
echo "=== How to Use ==="
echo ""
echo "1. Prepare a voice sample (10-30 seconds of clear audio)"
echo "   Example: /path/to/voice_sample.wav"
echo ""
echo "2. List available audio devices:"
echo "   uv run voice_clone.py --list-devices"
echo ""
echo "3. Run the voice cloner:"
echo "   uv run voice_clone.py /path/to/voice_sample.wav"
echo ""
echo "4. In Google Meet, Zoom, or other conferencing app:"
echo "   - Go to audio settings"
echo "   - Select 'Voice Clone Virtual Output' as your microphone"
echo "   - Start speaking!"
echo ""
echo "=== Quick Example ==="
echo ""
echo "  # Download a sample voice file (optional)"
echo "  wget https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3 -O sample.mp3"
echo ""
echo "  # Run with default settings"
echo "  uv run voice_clone.py sample.mp3"
echo ""
echo "For more options, see README.md"
echo ""
