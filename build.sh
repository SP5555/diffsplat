#!/bin/bash

# find nvcc wherever it lives
if command -v nvcc &>/dev/null; then
    echo "Using nvcc: $(which nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export PATH=/usr/local/cuda/bin:$PATH
    echo "Using nvcc: /usr/local/cuda/bin/nvcc"
else
    echo "ERROR: nvcc not found"
    exit 1
fi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)