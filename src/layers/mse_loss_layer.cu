#include "mse_loss_layer.h"
#include <cuda_runtime.h>
#include <stdexcept>

#include "../utils/cuda_utils.h"

/* ===== ===== Kernels ===== ===== */

/**
 * Computes per-pixel MSE gradient
 * 
 * dL/d_pixel_i = (2 / num_pixels) * (pixel_i - target_i)
 *
 * @param[in]  d_pixels     Rendered pixel colors [H*W*3]
 * @param[in]  d_target     Target pixel colors   [H*W*3]
 * @param[out] d_grad       dL/d_pixels           [H*W*3]
 * @param[in]  num_pixels   H * W
 */
__global__ void mseGradKernel(
    const float *__restrict__ d_pixels,
    const float *__restrict__ d_target,
    float *d_grad,
    size_t num_pixels)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pixels) return;

    float scale = 2.f / (float)num_pixels;
    d_grad[i * 3 + 0] = scale * (d_pixels[i * 3 + 0] - d_target[i * 3 + 0]);
    d_grad[i * 3 + 1] = scale * (d_pixels[i * 3 + 1] - d_target[i * 3 + 1]);
    d_grad[i * 3 + 2] = scale * (d_pixels[i * 3 + 2] - d_target[i * 3 + 2]);
}

/**
 * Computes per-pixel MSE and accumulates into a scalar.
 * Only called by forward() which is optional (logging only, not every frame).
 * 
 * L = (1 / num_pixels) * sum_i( (pixel_i - target_i)^2 )
 * 
 * @param[in]  d_pixels     Rendered pixel colors [H*W*3]
 * @param[in]  d_target     Target pixel colors   [H*W*3]
 * @param[out] d_loss       Scalar loss output    [1]
 * @param[in]  num_pixels   H * W
 */
__global__ void mseLossKernel(
    const float *__restrict__ d_pixels,
    const float *__restrict__ d_target,
    float *d_loss,
    size_t num_pixels)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pixels) return;

    float dr = d_pixels[i * 3 + 0] - d_target[i * 3 + 0];
    float dg = d_pixels[i * 3 + 1] - d_target[i * 3 + 1];
    float db = d_pixels[i * 3 + 2] - d_target[i * 3 + 2];
    float pixel_loss = (dr * dr + dg * dg + db * db) / (float)num_pixels;

    atomicAdd(d_loss, pixel_loss);
}

/* ===== ===== Lifecycle ===== ===== */

void MSELossLayer::allocate(int width, int height)
{
    num_pixels = width * height;
    d_grad_pixels.allocate(num_pixels * 3);
    d_loss.allocate(1);
}

void MSELossLayer::zero_grad()
{
    if (d_grad_pixels)
        cudaMemset(d_grad_pixels, 0, num_pixels * 3 * sizeof(float));
}

/* ===== ===== Forward / Backward ===== ===== */

void MSELossLayer::forward()
{
    int threads = 256;
    int blocks  = (num_pixels + threads - 1) / threads;

    cudaMemset(d_loss, 0, sizeof(float));
    mseLossKernel<<<blocks, threads>>>(input, d_target_pixels, d_loss, num_pixels);
    CUDA_SYNC_CHECK();
}

void MSELossLayer::backward()
{
    int threads = 256;
    int blocks  = (num_pixels + threads - 1) / threads;
    mseGradKernel<<<blocks, threads>>>(input, d_target_pixels, d_grad_pixels, num_pixels);
    CUDA_SYNC_CHECK();
}

float MSELossLayer::getLoss() const
{
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss;
}