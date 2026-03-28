#pragma once

// Threads per block for general-purpose 1D kernels.
#define BLOCK_SIZE 256

// Threads per block for the rasterize layer.
// CHUNK_SIZE must equal BLOCK_THREADS (one thread loads one splat per chunk).
#define BLOCK_THREADS 256
#define CHUNK_SIZE    256
