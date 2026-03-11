#include "ndc_project_layer.h"
#include <cuda_runtime.h>
#include <stdexcept>

#include "../utils/cuda_utils.cuh"

/* ===== ===== Kernels ===== ===== */

/**
 * Forward kernel: transforms world-space Gaussian params into NDC-space.
 * 
 * One thread is launched per splat.
 *
 * @param[in]  pos_x_world  World-space x positions [N]
 * @param[in]  pos_y_world  World-space y positions [N]
 * @param[in]  cov_a_world  World-space cov_a [N]
 * @param[in]  cov_b_world  World-space cov_b [N]
 * @param[in]  cov_d_world  World-space cov_d [N]
 * @param[out] pos_x_ndc    NDC-space x positions [N]
 * @param[out] pos_y_ndc    NDC-space y positions [N]
 * @param[out] cov_a_ndc    NDC-space cov_a [N]
 * @param[out] cov_b_ndc    NDC-space cov_b [N]
 * @param[out] cov_d_ndc    NDC-space cov_d [N]
 * @param[in]  sx           Scale factor x = 2 / width
 * @param[in]  sy           Scale factor y = 2 / height
 * @param[in]  count        Number of splats
 */
__global__ void ndcForwardKernel(
    const float *__restrict__ pos_x_world,
    const float *__restrict__ pos_y_world,
    const float *__restrict__ cov_a_world,
    const float *__restrict__ cov_b_world,
    const float *__restrict__ cov_d_world,
    float *pos_x_ndc,
    float *pos_y_ndc,
    float *cov_a_ndc,
    float *cov_b_ndc,
    float *cov_d_ndc,
    float sx, float sy,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    pos_x_ndc[i] = pos_x_world[i] * sx;
    pos_y_ndc[i] = pos_y_world[i] * sy;
    cov_a_ndc[i] = cov_a_world[i] * sx * sx;
    cov_b_ndc[i] = cov_b_world[i] * sx * sy;
    cov_d_ndc[i] = cov_d_world[i] * sy * sy;
}

/**
 * Backward kernel: chains gradients from NDC-space back to world-space.
 *
 * @param[in]  grad_pos_x_ndc   dL/d_pos_x_ndc [N]
 * @param[in]  grad_pos_y_ndc   dL/d_pos_y_ndc [N]
 * @param[in]  grad_cov_a_ndc   dL/d_cov_a_ndc [N]
 * @param[in]  grad_cov_b_ndc   dL/d_cov_b_ndc [N]
 * @param[in]  grad_cov_d_ndc   dL/d_cov_d_ndc [N]
 * @param[in]  grad_color_r_ndc dL/d_color_r   [N]
 * @param[in]  grad_color_g_ndc dL/d_color_g   [N]
 * @param[in]  grad_color_b_ndc dL/d_color_b   [N]
 * @param[in]  grad_opacity_ndc dL/d_opacity   [N]
 * @param[out] grad_pos_x_world dL/d_pos_x_world [N]
 * @param[out] grad_pos_y_world dL/d_pos_y_world [N]
 * @param[out] grad_cov_a_world dL/d_cov_a_world [N]
 * @param[out] grad_cov_b_world dL/d_cov_b_world [N]
 * @param[out] grad_cov_d_world dL/d_cov_d_world [N]
 * @param[out] grad_color_r     dL/d_color_r     [N]
 * @param[out] grad_color_g     dL/d_color_g     [N]
 * @param[out] grad_color_b     dL/d_color_b     [N]
 * @param[out] grad_opacity     dL/d_opacity     [N]
 * @param[in]  sx               Scale factor x = 2 / width
 * @param[in]  sy               Scale factor y = 2 / height
 * @param[in]  count            Number of splats
 */
__global__ void ndcBackwardKernel(
    const float *__restrict__ grad_pos_x_ndc,
    const float *__restrict__ grad_pos_y_ndc,
    const float *__restrict__ grad_cov_a_ndc,
    const float *__restrict__ grad_cov_b_ndc,
    const float *__restrict__ grad_cov_d_ndc,
    const float *__restrict__ grad_color_r_ndc,
    const float *__restrict__ grad_color_g_ndc,
    const float *__restrict__ grad_color_b_ndc,
    const float *__restrict__ grad_opacity_ndc,
    float *grad_pos_x_world,
    float *grad_pos_y_world,
    float *grad_cov_a_world,
    float *grad_cov_b_world,
    float *grad_cov_d_world,
    float *grad_color_r,
    float *grad_color_g,
    float *grad_color_b,
    float *grad_opacity,
    float sx, float sy,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    grad_pos_x_world[i] = grad_pos_x_ndc[i] * sx;
    grad_pos_y_world[i] = grad_pos_y_ndc[i] * sy;
    grad_cov_a_world[i] = grad_cov_a_ndc[i] * sx * sx;
    grad_cov_b_world[i] = grad_cov_b_ndc[i] * sx * sy;
    grad_cov_d_world[i] = grad_cov_d_ndc[i] * sy * sy;

    grad_color_r[i] = grad_color_r_ndc[i];
    grad_color_g[i] = grad_color_g_ndc[i];
    grad_color_b[i] = grad_color_b_ndc[i];
    grad_opacity[i] = grad_opacity_ndc[i];
}

/* ===== ===== Lifecycle ===== ===== */

void NDCProjectLayer::allocate(int width, int height, int count)
{
    screen_width = width;
    screen_height = height;
    allocatedCount = count;

    auto alloc = [](int n) {
        float *p = nullptr;
        cudaMalloc(&p, n * sizeof(float));
        return p;
    };

    output.pos_x = alloc(count);
    output.pos_y = alloc(count);
    output.cov_a = alloc(count);
    output.cov_b = alloc(count);
    output.cov_d = alloc(count);
    output.count = count;
}

void NDCProjectLayer::free()
{
    CUDA_FREE(output.pos_x);
    CUDA_FREE(output.pos_y);
    CUDA_FREE(output.cov_a);
    CUDA_FREE(output.cov_b);
    CUDA_FREE(output.cov_d);
    output.count   = 0;
    allocatedCount = 0;
}

void NDCProjectLayer::zero_grad()
{
    // this layer doesn't own the gradients
    // nothing to zero out
}

/* ===== ===== Forward / Backward ===== ===== */

void NDCProjectLayer::forward()
{
    float sx = 2.f / (float)screen_width;
    float sy = 2.f / (float)screen_height;
    int count = input->count;

    // wire pass-through aliases (no copy, just pointer assignment)
    output.pos_z   = input->pos_z;
    output.color_r = input->color_r;
    output.color_g = input->color_g;
    output.color_b = input->color_b;
    output.opacity = input->opacity;
    output.count   = count;

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    ndcForwardKernel<<<blocks, threads>>>(
        input->pos_x, input->pos_y,
        input->cov_a, input->cov_b, input->cov_d,
        output.pos_x, output.pos_y,
        output.cov_a, output.cov_b, output.cov_d,
        sx, sy, count
    );
    cudaDeviceSynchronize();
}

void NDCProjectLayer::backward()
{
    float sx = 2.f / (float)screen_width;
    float sy = 2.f / (float)screen_height;
    int count = input->count;

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    ndcBackwardKernel<<<blocks, threads>>>(
        gradOutput->pos_x,   gradOutput->pos_y,
        gradOutput->cov_a,   gradOutput->cov_b,   gradOutput->cov_d,
        gradOutput->color_r, gradOutput->color_g, gradOutput->color_b,
        gradOutput->opacity,
        gradInput->grad_pos_x, gradInput->grad_pos_y,
        gradInput->grad_cov_a, gradInput->grad_cov_b, gradInput->grad_cov_d,
        gradInput->grad_color_r, gradInput->grad_color_g, gradInput->grad_color_b,
        gradInput->grad_opacity,
        sx, sy, count
    );
    cudaDeviceSynchronize();
}
