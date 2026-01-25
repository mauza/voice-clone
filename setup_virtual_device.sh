#!/bin/bash
# Setup virtual audio device for voice cloning

set -e

echo "=== Voice Cloning Virtual Audio Device Setup ==="
echo ""

# Detect if using PulseAudio or PipeWire
if command -v pw-cli &> /dev/null; then
    AUDIO_SYSTEM="pipewire"
    echo "Detected: PipeWire"
elif command -v pactl &> /dev/null; then
    AUDIO_SYSTEM="pulseaudio"
    echo "Detected: PulseAudio"
else
    echo "Error: Neither PulseAudio nor PipeWire found!"
    exit 1
fi

echo ""

# Create virtual sink (output device)
SINK_NAME="voice_clone_output"
SINK_DESCRIPTION="Voice Clone Virtual Output"

if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
    echo "Creating PipeWire virtual sink..."
    
    # Check if module-null-sink is already loaded
    if pactl list sinks short | grep -q "$SINK_NAME"; then
        echo "Virtual sink already exists. Removing old one..."
        pactl unload-module module-null-sink
    fi
    
    # Create virtual sink
    pactl load-module module-null-sink \
        sink_name="$SINK_NAME" \
        sink_properties=device.description="$SINK_DESCRIPTION"
    
    echo "✓ Virtual sink created: $SINK_NAME"
    
elif [ "$AUDIO_SYSTEM" = "pulseaudio" ]; then
    echo "Creating PulseAudio virtual sink..."
    
    # Check if module already loaded
    if pactl list sinks short | grep -q "$SINK_NAME"; then
        echo "Virtual sink already exists. Removing old one..."
        pactl unload-module module-null-sink
    fi
    
    # Create virtual sink
    pactl load-module module-null-sink \
        sink_name="$SINK_NAME" \
        sink_properties=device.description="$SINK_DESCRIPTION"
    
    echo "✓ Virtual sink created: $SINK_NAME"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Virtual audio device '$SINK_DESCRIPTION' is now available!"
echo ""
echo "To use in conferencing apps (Google Meet, Zoom, etc.):"
echo "1. Run the voice cloning script with --output-device pointing to this sink"
echo "2. In your conferencing app, select '$SINK_DESCRIPTION' as your microphone"
echo "3. Speak into your real microphone"
echo "4. Your voice will be converted to the cloned voice"
echo ""
echo "To list all audio devices:"
echo "  python voice_clone.py --list-devices"
echo ""
echo "To remove the virtual device later:"
echo "  pactl unload-module module-null-sink"
echo ""
