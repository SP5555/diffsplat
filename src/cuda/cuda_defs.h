#pragma once

// Threads per block for general-purpose 1D kernels.
#define BLOCK_SIZE 256

// Ceiling integer division: divRoundUp(7, 4) == 2.
__host__ __device__ inline int divRoundUp(int n, int d) { return (n + d - 1) / d; }
