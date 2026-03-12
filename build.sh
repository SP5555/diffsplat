#!/bin/bash

# always add cuda to PATH first
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# now verify
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found even after adding /usr/local/cuda/bin to PATH"
    exit 1
fi

echo "Using nvcc: $(which nvcc)"
echo "nvcc version: $(nvcc --version | grep release)"

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)