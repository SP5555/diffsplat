#include <cuda_runtime.h>

__global__ void clampFKernel(
    float* data,
    float min_val,
    float max_val,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];
    val = fmaxf(val, min_val);
    val = fminf(val, max_val);
    data[idx] = val;
}

void launchClampF(
    float* data,
    float min_val,
    float max_val,
    int n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    clampFKernel<<<numBlocks, blockSize>>>(data, min_val, max_val, n);
}