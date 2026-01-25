#!/bin/bash
# Voice Cloning Virtual Audio Device - Runner
# Handles setup, virtual device creation, and running the voice cloner

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Show usage
show_help() {
    cat << 'EOF'
Voice Cloning Virtual Audio Device

USAGE:
  ./run.sh <command> [options]

 COMMANDS:
   setup                 Install dependencies and setup virtual device
   start <voice_sample>  Start voice cloning
   list                  List available audio devices
   mute                  Mute virtual sink (stops playing through speakers)
   benchmark             Check hardware capabilities
   remove                Remove virtual audio device
   help                  Show this help

SETUP:
  ./run.sh setup        # First time setup

EXAMPLES:
  # Setup (run once)
  ./run.sh setup

  # Start with voice transcript (recommended)
  ./run.sh start voice_sample.wav "Your transcript here"

  # Start with speaker embedding only (no transcript)
  ./run.sh start voice_sample.wav

   # List audio devices (to find the monitor source)
   ./run.sh list

   # Mute virtual sink if it's playing through speakers
   ./run.sh mute

   # Check hardware
   ./run.sh benchmark

  # Advanced usage - pass options directly
  ./run.sh start voice_sample.wav "Your transcript" --whisper-model medium
  ./run.sh start voice_sample.wav "" --x-vector-only

Voice Cloning (STT→TTS):
  - Latency: 2-5s
  - VRAM: 8-12GB
  - Transcript recommended (optional with --x-vector-only)
  - Converts speech to text, then synthesizes in target voice

For more info, see README.md
EOF
}

# Check system requirements
check_requirements() {
    local missing=0

    # Check for audio system
    if command -v pw-cli &> /dev/null; then
        AUDIO_SYSTEM="pipewire"
        print_success "Audio system: PipeWire"
    elif command -v pactl &> /dev/null; then
        AUDIO_SYSTEM="pulseaudio"
        print_success "Audio system: PulseAudio"
    else
        print_error "Neither PulseAudio nor PipeWire found!"
        echo "  Install one of these audio systems first."
        missing=1
    fi

    # Check for Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python 3 not found!"
        missing=1
    fi

    return $missing
}

# Setup everything
do_setup() {
    print_header "Voice Cloning Setup"

    # Check requirements
    if ! check_requirements; then
        echo ""
        print_error "Missing system requirements. Please install them first."
        exit 1
    fi

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo ""
        print_warning "uv package manager not found. Installing..."
        echo ""
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Source the shell to get uv in PATH
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            print_error "Failed to install uv. Please install manually:"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
    fi
    print_success "uv package manager installed"

    # Install Python dependencies
    echo ""
    print_header "Installing Python Dependencies"
    echo "This may take a few minutes..."
    uv sync
    print_success "Python dependencies installed"

    # Setup virtual audio device
    echo ""
    print_header "Setting Up Virtual Audio Device"
    setup_virtual_device

    # Check for GPU
    echo ""
    print_header "GPU Detection"
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        print_success "GPU detected: $GPU_NAME"
        echo ""
        echo "GPU support already installed!"
    else
        print_warning "No GPU detected or PyTorch not installed with GPU support"
   echo ""
   echo "Voice Cloning (STT→TTS):"
   echo "   - Latency: 2-5s"
   echo "   - VRAM: 8-12GB"
   echo "   - Transcript recommended (optional with --x-vector-only)"
   echo "   - Converts speech to text, then synthesizes in target voice"
   echo ""
   echo "   The output goes to a virtual sink. Use the monitor source as your"
   echo "   microphone input in Google Meet / Zoom / Discord."
   echo ""
        echo "  NVIDIA GPUs:"
        echo "    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo ""
        echo "  AMD GPUs:"
        echo "    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0"
        echo ""
    fi

    # Show next steps
    echo ""
    print_header "Setup Complete!"
    echo ""
  echo "Quick start:"
  echo "  1. Get a voice sample (10-30 seconds of clear audio)"
  echo "  2. Run: ./run.sh start your_voice_sample.wav \"Your transcript\""
  echo "  3. In your conferencing app (Google Meet/Zoom), select the virtual device as your microphone"
  echo "  4. Run './run.sh list | grep -i voice' to find the monitor device name"
  echo ""
    echo "Examples:"
    echo "  ./run.sh list                          # List audio devices"
    echo "  ./run.sh start voice.wav \" transcript\" # Start with transcript"
    echo "  ./run.sh start voice.wav               # Start with --x-vector-only"
    echo "  ./run.sh benchmark                     # Check hardware"
    echo ""
}

# Setup virtual audio device
mute_virtual_sink() {
    SINK_NAME="voice_clone_output"
    
    if pactl list sinks short 2>/dev/null | grep -q "$SINK_NAME"; then
        pactl set-sink-mute "$SINK_NAME" true 2>/dev/null || true
        print_success "Virtual sink muted - now use monitor source as microphone"
    else
        print_warning "Virtual sink not found. Run './run.sh setup' first."
    fi
}

setup_virtual_device() {
    SINK_NAME="voice_clone_output"
    SINK_DESCRIPTION="Voice Clone Virtual Output"

    # Check if already exists
    if pactl list sinks short 2>/dev/null | grep -q "$SINK_NAME"; then
        print_warning "Virtual sink already exists. Recreating..."
        pactl unload-module module-null-sink 2>/dev/null || true
        sleep 0.5
    fi

    # Create virtual sink
    if pactl load-module module-null-sink \
        sink_name="$SINK_NAME" \
        sink_properties=device.description="$SINK_DESCRIPTION" >/dev/null 2>&1; then
        print_success "Virtual audio device created: $SINK_DESCRIPTION"
        
        # Mute the virtual sink so it doesn't play through speakers
        pactl set-sink-mute "$SINK_NAME" true 2>/dev/null || true
        print_success "Virtual sink muted (use monitor source as microphone)"
    else
        print_error "Failed to create virtual audio device"
        exit 1
    fi
}

# Remove virtual audio device
remove_virtual_device() {
    print_header "Removing Virtual Audio Device"

    if pactl list sinks short 2>/dev/null | grep -q "voice_clone_output"; then
        pactl unload-module module-null-sink 2>/dev/null || true
        print_success "Virtual audio device removed"
    else
        print_warning "Virtual audio device not found (already removed?)"
    fi
}

# Start voice cloning
start_cloning() {
    VOICE_SAMPLE="$1"
    TRANSCRIPT="$2"
    shift 2 || shift

    if [ -z "$VOICE_SAMPLE" ]; then
        print_error "Voice sample file required!"
        echo ""
        echo "Usage: ./run.sh start <voice_sample> [transcript] [options]"
        echo ""
        echo "Example:"
        echo "  ./run.sh start voice.wav \"Your transcript here\""
        echo "  ./run.sh start voice.wav \"\" --x-vector-only"
        exit 1
    fi

    if [ ! -f "$VOICE_SAMPLE" ]; then
        print_error "Voice sample not found: $VOICE_SAMPLE"
        exit 1
    fi

    print_header "Starting Voice Cloning"
    echo "Voice sample: $VOICE_SAMPLE"
    if [ -n "$TRANSCRIPT" ]; then
        echo "Transcript: $TRANSCRIPT"
    else
        echo "Using --x-vector-only (no transcript)"
    fi
    echo "Mode: STT→TTS Pipeline"
    echo "Expected latency: 2-5s"
    echo ""

    # Set voice_clone_output as default sink for audio routing
    if pactl list sinks short | grep -q "voice_clone_output"; then
        PREVIOUS_SINK=$(pactl get-default-sink)
        pactl set-default-sink voice_clone_output 2>/dev/null || true
        print_success "Set voice_clone_output as default audio sink"

        # Restore previous sink on exit
        trap "pactl set-default-sink '$PREVIOUS_SINK' 2>/dev/null || true" EXIT
    else
        print_warning "Virtual sink not found. Run './run.sh setup' first."
    fi

    # Build command
    if [ -n "$TRANSCRIPT" ]; then
        uv run voice_clone.py "$VOICE_SAMPLE" --voice-text "$TRANSCRIPT" "$@"
    else
        uv run voice_clone.py "$VOICE_SAMPLE" --x-vector-only "$@"
    fi
}

# List audio devices
list_devices() {
    print_header "Available Audio Devices"
    uv run voice_clone.py dummy.wav --list-devices 2>/dev/null || \
        python3 -c "import sounddevice as sd; devices = sd.query_devices(); [print(f'{i}: {d[\"name\"]}') for i, d in enumerate(devices)]"

    echo ""
    if command -v pactl &> /dev/null; then
        print_header "Virtual Device Monitor (Use as Mic Input)"
        echo "Look for the '.monitor' source below - this is what you select as your microphone:"
        echo ""
        pactl list sources short | grep -i "voice_clone" && echo "" || echo "No virtual device found. Run './run.sh setup' first."
        
        # Show monitor details
        if pactl list sources short | grep -qi "voice_clone"; then
            echo "If you hear audio from your speakers, run: ./run.sh mute"
        fi
    fi
}

# Run benchmark
run_benchmark() {
    print_header "Hardware Benchmark"
    if [ -f "benchmark.py" ]; then
        uv run benchmark.py "$@"
    else
        print_error "benchmark.py not found"
        exit 1
    fi
}

# Main command handling
COMMAND="${1:-help}"

case "$COMMAND" in
    setup)
        do_setup
        ;;

    start)
        shift
        start_cloning "$@"
        ;;

    list)
        list_devices
        ;;

    mute)
        mute_virtual_sink
        ;;

    benchmark)
        shift
        run_benchmark "$@"
        ;;

    remove)
        remove_virtual_device
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        echo "Run './run.sh help' for usage information"
        exit 1
        ;;
esac
