#!/bin/bash

# script to force running on NVIDIA GPU if available
# might not work in all environments

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <app> [args...]"
    echo "  ./run.sh imgfitapp --width 1280 --height 720 --image path/to/image.png --splats-count 80000"
    echo "  ./run.sh plyviewapp --scene path/to/scene.ply --scale 1.0"
    exit 1
fi

APP="./build/$1"
shift  # remove app name, rest is passed through as args

if [ ! -x "$APP" ]; then
    echo "ERROR: $APP not found or not executable. Did you build first?"
    exit 1
fi

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia "$APP" "$@"