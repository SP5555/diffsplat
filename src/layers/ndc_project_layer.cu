#include "ndc_project_layer.h"
#include <cuda_runtime.h>

#include "../utils/cuda_utils.cuh"

/* ===== ===== Kernels ===== ===== */

/**
 * Forward kernel: projects world-space positions and 3D covariance
 * into 2D NDC space using an orthographic projection.
 * 
 * pos_ndc  = pos_world * s
 * J        = [[sx, 0, 0], [0, sy, 0]]
 * Cov2D    = J * Cov3D * J^T
 * 
 *   cov_xx_2d = sx*sx * cov_xx_3d
 *   cov_xy_2d = sx*sy * cov_xy_3d
 *   cov_yy_2d = sy*sy * cov_yy_3d
 * 
 * The Z row/column of Cov3D is dropped by the projection.
 * 
 * One thread is launched per splat.
 */
__global__ void ndcForwardKernel(
    // inputs: world space
    const float *__restrict__ pos_x_world,
    const float *__restrict__ pos_y_world,
    const float *__restrict__ cov_xx_3d,
    const float *__restrict__ cov_xy_3d,
    const float *__restrict__ cov_yy_3d,
    // outputs: NDC space
    float *pos_x_ndc,
    float *pos_y_ndc,
    float *cov_xx_2d,
    float *cov_xy_2d,
    float *cov_yy_2d,
    // projection scales
    float sx, float sy,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    pos_x_ndc[i] = pos_x_world[i] * sx;
    pos_y_ndc[i] = pos_y_world[i] * sy;

    cov_xx_2d[i] = cov_xx_3d[i] * sx * sx;
    cov_xy_2d[i] = cov_xy_3d[i] * sx * sy;
    cov_yy_2d[i] = cov_yy_3d[i] * sy * sy;
}

/**
 * Backward kernel: chains gradients from 2D NDC space back to 3D world space.
 * 
 * Chain rule through Cov2D = J * Cov3D * J^T gives:
 *   dL/dcov_xx_3d = dL/dcov_xx_2d * sx*sx
 *   dL/dcov_xy_3d = dL/dcov_xy_2d * sx*sy
 *   dL/dcov_yy_3d = dL/dcov_yy_2d * sy*sy
 *   dL/dcov_xz_3d = 0  (Z components dropped by projection)
 *   dL/dcov_yz_3d = 0
 *   dL/dcov_zz_3d = 0
 * 
 * pos_z gradient is zero (Z not transformed).
 * 
 * One thread is launched per splat.
 */
__global__ void ndcBackwardKernel(
    // grad_output
    const float *__restrict__ grad_out_pos_x,
    const float *__restrict__ grad_out_pos_y,
    const float *__restrict__ grad_out_cov_xx,
    const float *__restrict__ grad_out_cov_xy,
    const float *__restrict__ grad_out_cov_yy,
    // grad_input
    float *grad_in_pos_x,
    float *grad_in_pos_y,
    float *grad_in_cov_xx,
    float *grad_in_cov_xy,
    float *grad_in_cov_yy,
    // projection scales
    float sx, float sy,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    grad_in_pos_x[i] = grad_out_pos_x[i] * sx;
    grad_in_pos_y[i] = grad_out_pos_y[i] * sy;

    grad_in_cov_xx[i] = grad_out_cov_xx[i] * sx * sx;
    grad_in_cov_xy[i] = grad_out_cov_xy[i] * sx * sy;
    grad_in_cov_yy[i] = grad_out_cov_yy[i] * sy * sy;
}

/* ===== ===== Lifecycle ===== ===== */

void NDCProjectLayer::allocate(int width, int height, int count)
{
    screen_width  = width;
    screen_height = height;
    allocatedCount = count;
    output.allocate(count);
    gradInput.allocate(count);
}

void NDCProjectLayer::zero_grad()
{
    gradInput.zero_grad();
}

/* ===== ===== Forward / Backward ===== ===== */

void NDCProjectLayer::forward()
{
    float sx = 2.f / (float)screen_width;
    float sy = 2.f / (float)screen_height;
    int count = input->count;

    // pass-throughs
    cudaMemcpy(output.pos_z,   input->pos_z,   count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.color_r, input->color_r, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.color_g, input->color_g, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.color_b, input->color_b, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.opacity, input->opacity, count * sizeof(float), cudaMemcpyDeviceToDevice);

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    ndcForwardKernel<<<blocks, threads>>>(
        input->pos_x,   input->pos_y,
        input->cov_xx,  input->cov_xy,  input->cov_yy,
        output.pos_x,   output.pos_y,
        output.cov_xx,  output.cov_xy,  output.cov_yy,
        sx, sy, count
    );
    cudaDeviceSynchronize();
}

void NDCProjectLayer::backward()
{
    float sx = 2.f / (float)screen_width;
    float sy = 2.f / (float)screen_height;
    int count = input->count;

    // pass-throughs
    cudaMemcpy(gradInput.grad_color_r, gradOutput->grad_color_r, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gradInput.grad_color_g, gradOutput->grad_color_g, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gradInput.grad_color_b, gradOutput->grad_color_b, count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gradInput.grad_opacity, gradOutput->grad_opacity, count * sizeof(float), cudaMemcpyDeviceToDevice);
    // pos_z gradient is zero since Z is not transformed by projection
    cudaMemset(gradInput.grad_pos_z, 0, count * sizeof(float));
    // cov_xz, cov_yz, cov_zz gradients are zero since Z components are dropped by projection
    cudaMemset(gradInput.grad_cov_xz, 0, count * sizeof(float));
    cudaMemset(gradInput.grad_cov_yz, 0, count * sizeof(float));
    cudaMemset(gradInput.grad_cov_zz, 0, count * sizeof(float));

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    ndcBackwardKernel<<<blocks, threads>>>(
        gradOutput->grad_pos_x,  gradOutput->grad_pos_y,
        gradOutput->grad_cov_xx, gradOutput->grad_cov_xy, gradOutput->grad_cov_yy,

        gradInput.grad_pos_x,
        gradInput.grad_pos_y,
        gradInput.grad_cov_xx, gradInput.grad_cov_xy, gradInput.grad_cov_yy,

        sx, sy, count
    );
    cudaDeviceSynchronize();
}
