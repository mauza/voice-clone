#!/bin/bash
# Voice Cloning Virtual Audio Device - Unified Runner
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
Voice Cloning Virtual Audio Device - Unified Runner

USAGE:
  ./run.sh <command> [options]

COMMANDS:
  setup                 Install dependencies and setup virtual device
  start <voice_sample>  Start voice cloning (fast mode)
  quality <voice_sample> <transcript>
                        Start voice cloning (quality mode)
  list                  List available audio devices
  benchmark             Check hardware capabilities
  remove                Remove virtual audio device
  help                  Show this help

SETUP:
  ./run.sh setup        # First time setup

EXAMPLES:
  # Setup (run once)
  ./run.sh setup

  # Start with fast mode (default)
  ./run.sh start voice_sample.wav

  # Start with quality mode
  ./run.sh quality voice_sample.wav "Your transcript here"

  # List audio devices
  ./run.sh list

  # Check hardware
  ./run.sh benchmark

  # Advanced usage - pass options directly
  ./run.sh start voice_sample.wav --converter rvc --model-path model.pth
  ./run.sh start voice_sample.wav --chunk-duration 0.3 --device cuda

MODES:
  Fast mode (default):
    - Latency: 100-300ms
    - VRAM: 2-4GB
    - No transcript needed
    - Use for: video calls, gaming, streaming

  Quality mode:
    - Latency: 2-5s
    - VRAM: 8-12GB
    - Requires transcript
    - Use for: pre-recorded content, best quality

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
        echo "To enable GPU support (STRONGLY RECOMMENDED):"
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
    echo "  2. Run: ./run.sh start your_voice_sample.wav"
    echo "  3. In your conferencing app, select 'Voice Clone Virtual Output' as microphone"
    echo ""
    echo "Examples:"
    echo "  ./run.sh list                          # List audio devices"
    echo "  ./run.sh start voice.wav               # Start with fast mode"
    echo "  ./run.sh quality voice.wav \"text\"      # Start with quality mode"
    echo "  ./run.sh benchmark                     # Check hardware"
    echo ""
    echo "Optional: For best quality voice conversion, install RVC:"
    echo "  uv pip install rvc-python faiss-cpu praat-parselmouth"
    echo "  (Then train an RVC model - see README.md for instructions)"
    echo ""
}

# Setup virtual audio device
setup_virtual_device() {
    SINK_NAME="voice_clone_output"
    SINK_DESCRIPTION="Voice Clone Virtual Output"

    # Check if already exists
    if pactl list sinks short 2>/dev/null | grep -q "$SINK_NAME"; then
        print_warning "Virtual sink already exists. Recreating..."
        pactl unload-module module-null-sink 2>/dev/null || true
    fi

    # Create virtual sink
    if pactl load-module module-null-sink \
        sink_name="$SINK_NAME" \
        sink_properties=device.description="$SINK_DESCRIPTION" >/dev/null 2>&1; then
        print_success "Virtual audio device created: $SINK_DESCRIPTION"
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

# Start voice cloning (fast mode)
start_fast() {
    VOICE_SAMPLE="$1"
    shift

    if [ -z "$VOICE_SAMPLE" ]; then
        print_error "Voice sample file required!"
        echo ""
        echo "Usage: ./run.sh start <voice_sample> [options]"
        echo ""
        echo "Example:"
        echo "  ./run.sh start voice.wav"
        echo "  ./run.sh start voice.wav --converter rvc --model-path model.pth"
        exit 1
    fi

    if [ ! -f "$VOICE_SAMPLE" ]; then
        print_error "Voice sample not found: $VOICE_SAMPLE"
        exit 1
    fi

    print_header "Starting Voice Cloning (Fast Mode)"
    echo "Voice sample: $VOICE_SAMPLE"
    echo "Mode: Fast (Voice Conversion)"
    echo "Expected latency: 100-300ms"
    echo ""

    uv run voice_clone.py "$VOICE_SAMPLE" --mode fast "$@"
}

# Start voice cloning (quality mode)
start_quality() {
    VOICE_SAMPLE="$1"
    TRANSCRIPT="$2"
    shift 2

    if [ -z "$VOICE_SAMPLE" ] || [ -z "$TRANSCRIPT" ]; then
        print_error "Voice sample and transcript required!"
        echo ""
        echo "Usage: ./run.sh quality <voice_sample> <transcript> [options]"
        echo ""
        echo "Example:"
        echo "  ./run.sh quality voice.wav \"This is my voice sample\""
        exit 1
    fi

    if [ ! -f "$VOICE_SAMPLE" ]; then
        print_error "Voice sample not found: $VOICE_SAMPLE"
        exit 1
    fi

    print_header "Starting Voice Cloning (Quality Mode)"
    echo "Voice sample: $VOICE_SAMPLE"
    echo "Transcript: $TRANSCRIPT"
    echo "Mode: Quality (STT→TTS)"
    echo "Expected latency: 2-5s"
    echo ""

    uv run voice_clone.py "$VOICE_SAMPLE" --mode quality --voice-text "$TRANSCRIPT" "$@"
}

# List audio devices
list_devices() {
    print_header "Available Audio Devices"
    uv run voice_clone.py voice_sample.wav --list-devices 2>/dev/null || \
        python3 -c "import sounddevice as sd; devices = sd.query_devices(); [print(f'{i}: {d[\"name\"]}') for i, d in enumerate(devices)]"
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

# Transcribe audio file
transcribe_file() {
    AUDIO_FILE="$1"

    if [ -z "$AUDIO_FILE" ]; then
        print_error "Audio file required!"
        echo ""
        echo "Usage: ./run.sh transcribe <audio_file>"
        exit 1
    fi

    if [ ! -f "$AUDIO_FILE" ]; then
        print_error "Audio file not found: $AUDIO_FILE"
        exit 1
    fi

    print_header "Transcribing Audio"
    print_warning "Transcription feature removed. Use quality mode with --x-vector-only instead."
    echo ""
    echo "Or manually transcribe with Whisper:"
    echo "  uv pip install faster-whisper"
    echo "  python -c \"from faster_whisper import WhisperModel; m = WhisperModel('base'); segs, _ = m.transcribe('$AUDIO_FILE'); print(' '.join([s.text for s in segs]))\""
}

# Main command handling
COMMAND="${1:-help}"

case "$COMMAND" in
    setup)
        do_setup
        ;;

    start)
        shift
        start_fast "$@"
        ;;

    quality)
        shift
        start_quality "$@"
        ;;

    list)
        list_devices
        ;;

    benchmark)
        shift
        run_benchmark "$@"
        ;;

    transcribe)
        shift
        transcribe_file "$@"
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
