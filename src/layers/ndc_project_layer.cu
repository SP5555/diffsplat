#include "ndc_project_layer.h"
#include <cuda_runtime.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

// regularization for covariance gradients 
#define COV_L2_REG      1e-8f // so small? don't ask. fine-hand-tuned

/* ===== ===== Kernels ===== ===== */

/**
 * Forward kernel: projects world-space positions and 3D covariance
 * into 2D NDC space using an orthographic projection.
 * 
 * pos_ndc  = pos_world * s   (sz = 2 / min(width, height) for Z)
 * J        = [[sx, 0, 0], [0, sy, 0]]   (Z row dropped for 2D covariance)
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
    // inputs (world space)
    const float *__restrict__ i_w_x,
    const float *__restrict__ i_w_y,
    const float *__restrict__ i_w_z,
    const float *__restrict__ i_w_cxx,
    const float *__restrict__ i_w_cxy,
    const float *__restrict__ i_w_cyy,
    const float *__restrict__ i_R,
    const float *__restrict__ i_G,
    const float *__restrict__ i_B,
    const float *__restrict__ i_A,
    // outputs (NDC space)
    float *o_ndc_x,
    float *o_ndc_y,
    float *o_ndc_z,
    float *o_ndc_cxx,
    float *o_ndc_cxy,
    float *o_ndc_cyy,
    float *o_R,
    float *o_G,
    float *o_B,
    float *o_A,
    // projection scales
    float sx, float sy, float sz,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    o_ndc_x[i] = i_w_x[i] * sx;
    o_ndc_y[i] = i_w_y[i] * sy;
    o_ndc_z[i] = i_w_z[i] * sz;

    o_ndc_cxx[i] = i_w_cxx[i] * sx * sx;
    o_ndc_cxy[i] = i_w_cxy[i] * sx * sy;
    o_ndc_cyy[i] = i_w_cyy[i] * sy * sy;

    // pass-throughs
    o_R[i] = i_R[i];
    o_G[i] = i_G[i];
    o_B[i] = i_B[i];
    o_A[i] = i_A[i];
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
    // gradient output
    const float *__restrict__ grad_o_w_x,
    const float *__restrict__ grad_o_w_y,
    const float *__restrict__ grad_o_w_cxx,
    const float *__restrict__ grad_o_w_cxy,
    const float *__restrict__ grad_o_w_cyy,
    const float *__restrict__ grad_o_R,
    const float *__restrict__ grad_o_G,
    const float *__restrict__ grad_o_B,
    const float *__restrict__ grad_o_A,
    // gradient input
    float *grad_i_ndc_x,
    float *grad_i_ndc_y,
    float *grad_i_ndc_z,
    float *grad_i_ndc_cxx,
    float *grad_i_ndc_cxy,
    float *grad_i_ndc_cxz,
    float *grad_i_ndc_cyy,
    float *grad_i_ndc_cyz,
    float *grad_i_ndc_czz,
    float *grad_i_R,
    float *grad_i_G,
    float *grad_i_B,
    float *grad_i_A,
    // projection scales (sz unused: Z is not differentiated)
    float sx, float sy,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    grad_i_ndc_x[i] = grad_o_w_x[i] * sx;
    grad_i_ndc_y[i] = grad_o_w_y[i] * sy;
    grad_i_ndc_z[i] = 0;

    grad_i_ndc_cxx[i] = grad_o_w_cxx[i] * sx * sx;
    grad_i_ndc_cxy[i] = grad_o_w_cxy[i] * sx * sy;
    grad_i_ndc_cxz[i] = 0;
    grad_i_ndc_cyy[i] = grad_o_w_cyy[i] * sy * sy;
    grad_i_ndc_cyz[i] = 0;
    grad_i_ndc_czz[i] = 0;

    // pass-throughs
    grad_i_R[i] = grad_o_R[i];
    grad_i_G[i] = grad_o_G[i];
    grad_i_B[i] = grad_o_B[i];
    grad_i_A[i] = grad_o_A[i];
}

// ===== ===== Covariance Regularization Kernel ===== =====
/**
 * Applies L2 regularization to the covariance gradients.
 * Only off-diagonal entries are regularized -- they drive the most extreme
 * stretching (splat thin in one axis, elongated in the other) and benefit
 * most from a small penalty. Diagonal entries are left unpenalized.
 */
__global__ void covRegKernel(
    const float *__restrict__ cxx,
    const float *__restrict__ cxy,
    const float *__restrict__ cxz,
    const float *__restrict__ cyy,
    const float *__restrict__ cyz,
    const float *__restrict__ czz,
    float *grad_cxx,
    float *grad_cxy,
    float *grad_cxz,
    float *grad_cyy,
    float *grad_cyz,
    float *grad_czz,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // Diagonal entries (cxx, cyy, czz / grad_cxx, grad_cyy, grad_czz) are
    // intentionally not regularized -- add here if needed in the future.
    grad_cxy[i] += COV_L2_REG * (2.f * cxy[i]);
    grad_cxz[i] += COV_L2_REG * (2.f * cxz[i]);
    grad_cyz[i] += COV_L2_REG * (2.f * cyz[i]);
}

/* ===== ===== Lifecycle ===== ===== */

void NDCProjectLayer::allocate(int width, int height, int count)
{
    screen_width  = width;
    screen_height = height;
    allocated_count = count;
    out.allocate(count);
    grad_in.allocate(count);
}

void NDCProjectLayer::zero_grad()
{
    grad_in.zero_grad();
}

/* ===== ===== Forward / Backward ===== ===== */

void NDCProjectLayer::forward()
{
    float sx = 2.f / (float)screen_width;
    float sy = 2.f / (float)screen_height;
    float sz = 2.f / (float)(screen_width < screen_height ? screen_width : screen_height);
    int count = in->count;

    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ndcForwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->pos_x,   in->pos_y,   in->pos_z,
        in->cov_xx,  in->cov_xy,  in->cov_yy,
        in->color_r, in->color_g, in->color_b, in->opacity,
        out.pos_x,   out.pos_y,   out.pos_z,
        out.cov_xx,  out.cov_xy,  out.cov_yy,
        out.color_r, out.color_g, out.color_b, out.opacity,
        sx, sy, sz, count
    );
    CUDA_SYNC_CHECK();
}

void NDCProjectLayer::backward()
{
    float sx = 2.f / (float)screen_width;
    float sy = 2.f / (float)screen_height;
    int count = in->count;
    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    {
        ndcBackwardKernel<<<blocks, BLOCK_SIZE>>>(
            grad_out->grad_pos_x,   grad_out->grad_pos_y,
            grad_out->grad_cov_xx,  grad_out->grad_cov_xy,  grad_out->grad_cov_yy,
            grad_out->grad_color_r, grad_out->grad_color_g, grad_out->grad_color_b,
            grad_out->grad_opacity,
            grad_in.grad_pos_x,   grad_in.grad_pos_y,   grad_in.grad_pos_z,
            grad_in.grad_cov_xx,  grad_in.grad_cov_xy,  grad_in.grad_cov_xz,
            grad_in.grad_cov_yy,  grad_in.grad_cov_yz,  grad_in.grad_cov_zz,
            grad_in.grad_color_r, grad_in.grad_color_g, grad_in.grad_color_b,
            grad_in.grad_opacity,
            sx, sy, count
        );
        CUDA_SYNC_CHECK();
    }

    // covariance regularization
    {
        covRegKernel<<<blocks, BLOCK_SIZE>>>(
            in->cov_xx, in->cov_xy, in->cov_xz,
            in->cov_yy, in->cov_yz, in->cov_zz,
            grad_in.grad_cov_xx, grad_in.grad_cov_xy, grad_in.grad_cov_xz,
            grad_in.grad_cov_yy, grad_in.grad_cov_yz, grad_in.grad_cov_zz,
            count
        );
        CUDA_SYNC_CHECK();
    }
}
