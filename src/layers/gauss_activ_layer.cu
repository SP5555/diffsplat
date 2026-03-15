#include "gauss_activ_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#include "../utils/cuda_utils.h"

static constexpr float C0 = 0.28209f;  // DC SH coefficient

/* ===== ===== Kernels ===== ===== */

/**
 * Forward kernel: computes 3D covariance Cov = M*M^T where M = R*S.
 *
 * Quaternion is normalized in-kernel for robustness.
 * Scale is exponentiated: s_actual = exp(s_log).
 * Opacity is passed through sigmoid: opacity = 1 / (1 + exp(-logit_opacity)).
 *
 * R from unit quaternion q = (w, x, y, z):
 *   R = | 1-2(yy+zz)   2(xy-wz)   2(xz+wy) |
 *       |   2(xy+wz) 1-2(xx+zz)   2(yz-wx) |
 *       |   2(xz-wy)   2(yz+wx) 1-2(xx+yy) |
 *
 * Cov = R * S * S^T * R^T  (symmetric, stored as upper triangle)
 *
 * One thread is launched per splat.
 */
__global__ void covForwardKernel(
    // inputs: log-scale, raw quaternion, logit-opacity
    const float *__restrict__ scale_x,
    const float *__restrict__ scale_y,
    const float *__restrict__ scale_z,
    const float *__restrict__ rot_w,
    const float *__restrict__ rot_x,
    const float *__restrict__ rot_y,
    const float *__restrict__ rot_z,
    const float *__restrict__ color_DC_SH_r,
    const float *__restrict__ color_DC_SH_g,
    const float *__restrict__ color_DC_SH_b,
    const float *__restrict__ logit_opacity,
    // outputs: 3D covariance upper triangle + activated opacity
    float *cov_xx,      float *cov_xy,      float *cov_xz,
    float *cov_yy,      float *cov_yz,      float *cov_zz,
    float *color_lin_r, float *color_lin_g, float *color_lin_b,
    float *opacity,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // normalize quaternion
    float qw = rot_w[i], qx = rot_x[i], qy = rot_y[i], qz = rot_z[i];
    float norm = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    qw *= norm; qx *= norm; qy *= norm; qz *= norm;

    // actual scale
    float sx = expf(scale_x[i]);
    float sy = expf(scale_y[i]);
    float sz = expf(scale_z[i]);

    // rotation matrix columns (R[:,0], R[:,1], R[:,2])
    float r00 = 1.f - 2.f*(qy*qy + qz*qz);
    float r10 =       2.f*(qx*qy + qw*qz);
    float r20 =       2.f*(qx*qz - qw*qy);

    float r01 =       2.f*(qx*qy - qw*qz);
    float r11 = 1.f - 2.f*(qx*qx + qz*qz);
    float r21 =       2.f*(qy*qz + qw*qx);

    float r02 =       2.f*(qx*qz + qw*qy);
    float r12 =       2.f*(qy*qz - qw*qx);
    float r22 = 1.f - 2.f*(qx*qx + qy*qy);

    // M = R * S
    float m00 = r00 * sx,  m01 = r01 * sy,  m02 = r02 * sz;
    float m10 = r10 * sx,  m11 = r11 * sy,  m12 = r12 * sz;
    float m20 = r20 * sx,  m21 = r21 * sy,  m22 = r22 * sz;

    // Cov = M * M^T
    cov_xx[i] = m00*m00 + m01*m01 + m02*m02;
    cov_xy[i] = m00*m10 + m01*m11 + m02*m12;
    cov_xz[i] = m00*m20 + m01*m21 + m02*m22;
    cov_yy[i] = m10*m10 + m11*m11 + m12*m12;
    cov_yz[i] = m10*m20 + m11*m21 + m12*m22;
    cov_zz[i] = m20*m20 + m21*m21 + m22*m22;

    // DC SH coefficient -> linear RGB
    color_lin_r[i] = fminf(fmaxf(color_DC_SH_r[i] * C0 + 0.5f, 0.f), 1.f);
    color_lin_g[i] = fminf(fmaxf(color_DC_SH_g[i] * C0 + 0.5f, 0.f), 1.f);
    color_lin_b[i] = fminf(fmaxf(color_DC_SH_b[i] * C0 + 0.5f, 0.f), 1.f);

    // sigmoid activation on opacity
    opacity[i] = 1.f / (1.f + expf(-logit_opacity[i]));
}

/**
 * Backward kernel: chains dL/dCov back to log-scale and quaternion gradients,
 * and dL/dopacity back through sigmoid to logit-opacity.
 *
 * Given dL/dCov (upper triangle), chain rule through Cov = R*S*S^T*R^T.
 *
 *   dL/dM = 2 * dL/dCov_sym * M      (where dL/dCov_sym symmetrizes the grad)
 *   dL/dS = diag(R^T * dL/dM)        (gradient w.r.t. scale)
 *   dL/ds_log_i = dL/dS_i * exp(s_i) (chain through exp)
 *   dL/dR = dL/dM * S
 *   dL/dq = chain through R(q), then project through normalization Jacobian
 *
 * Sigmoid backward:
 *   s = sigmoid(logit),  dL/dlogit = dL/ds * s * (1 - s)
 *
 * One thread is launched per splat.
 */
__global__ void covBackwardKernel(
    // inputs: log-scale and raw quaternion (same as forward)
    const float *__restrict__ scale_x,
    const float *__restrict__ scale_y,
    const float *__restrict__ scale_z,
    const float *__restrict__ rot_w,
    const float *__restrict__ rot_x,
    const float *__restrict__ rot_y,
    const float *__restrict__ rot_z,
    const float *__restrict__ opacity,
    // grad_output: dL/dΣ upper triangle and dL/dopacity
    const float *__restrict__ grad_cov_xx,
    const float *__restrict__ grad_cov_xy,
    const float *__restrict__ grad_cov_xz,
    const float *__restrict__ grad_cov_yy,
    const float *__restrict__ grad_cov_yz,
    const float *__restrict__ grad_cov_zz,
    const float *__restrict__ grad_color_lin_r,
    const float *__restrict__ grad_color_lin_g,
    const float *__restrict__ grad_color_lin_b,
    const float *__restrict__ grad_opacity,
    // grad_input: gradients w.r.t. log-scale, raw quaternion, logit-opacity
    float *grad_scale_x,
    float *grad_scale_y,
    float *grad_scale_z,
    float *grad_rot_w,
    float *grad_rot_x,
    float *grad_rot_y,
    float *grad_rot_z,
    float *grad_color_DC_SH_r,
    float *grad_color_DC_SH_g,
    float *grad_color_DC_SH_b,
    float *grad_logit_opacity,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // --- recompute forward values ---
    float qw_raw = rot_w[i], qx_raw = rot_x[i], qy_raw = rot_y[i], qz_raw = rot_z[i];
    float norm_inv = rsqrtf(qw_raw*qw_raw + qx_raw*qx_raw + qy_raw*qy_raw + qz_raw*qz_raw);
    float qw = qw_raw * norm_inv;
    float qx = qx_raw * norm_inv;
    float qy = qy_raw * norm_inv;
    float qz = qz_raw * norm_inv;

    float sx = expf(scale_x[i]);
    float sy = expf(scale_y[i]);
    float sz = expf(scale_z[i]);

    // rotation matrix
    float r00 = 1.f - 2.f*(qy*qy + qz*qz);
    float r10 =       2.f*(qx*qy + qw*qz);
    float r20 =       2.f*(qx*qz - qw*qy);
    float r01 =       2.f*(qx*qy - qw*qz);
    float r11 = 1.f - 2.f*(qx*qx + qz*qz);
    float r21 =       2.f*(qy*qz + qw*qx);
    float r02 =       2.f*(qx*qz + qw*qy);
    float r12 =       2.f*(qy*qz - qw*qx);
    float r22 = 1.f - 2.f*(qx*qx + qy*qy);

    // M = R * S
    float m00 = r00*sx, m01 = r01*sy, m02 = r02*sz;
    float m10 = r10*sx, m11 = r11*sy, m12 = r12*sz;
    float m20 = r20*sx, m21 = r21*sy, m22 = r22*sz;

    // --- dL/dCov ---
    float dxx = grad_cov_xx[i];
    float dxy = grad_cov_xy[i];
    float dxz = grad_cov_xz[i];
    float dyy = grad_cov_yy[i];
    float dyz = grad_cov_yz[i];
    float dzz = grad_cov_zz[i];

    // --- dL/dM = 2 * dL/dCov_sym * M ---
    float dm00 = 2.f*(dxx*m00 + dxy*m10 + dxz*m20);
    float dm01 = 2.f*(dxx*m01 + dxy*m11 + dxz*m21);
    float dm02 = 2.f*(dxx*m02 + dxy*m12 + dxz*m22);
    float dm10 = 2.f*(dxy*m00 + dyy*m10 + dyz*m20);
    float dm11 = 2.f*(dxy*m01 + dyy*m11 + dyz*m21);
    float dm12 = 2.f*(dxy*m02 + dyy*m12 + dyz*m22);
    float dm20 = 2.f*(dxz*m00 + dyz*m10 + dzz*m20);
    float dm21 = 2.f*(dxz*m01 + dyz*m11 + dzz*m21);
    float dm22 = 2.f*(dxz*m02 + dyz*m12 + dzz*m22);

    // --- dL/ds_log ---
    grad_scale_x[i] = (dm00*r00 + dm10*r10 + dm20*r20) * sx;
    grad_scale_y[i] = (dm01*r01 + dm11*r11 + dm21*r21) * sy;
    grad_scale_z[i] = (dm02*r02 + dm12*r12 + dm22*r22) * sz;

    // --- dL/dR = dL/dM * S ---
    float dr00 = dm00*sx, dr10 = dm10*sx, dr20 = dm20*sx;
    float dr01 = dm01*sy, dr11 = dm11*sy, dr21 = dm21*sy;
    float dr02 = dm02*sz, dr12 = dm12*sz, dr22 = dm22*sz;

    // --- dL/dq_norm: chain through R(q) ---
    float dqw =  2.f*(
        dr10*qz - dr20*qy +
        dr01*(-qz) + dr21*qx +
        dr02*qy + dr12*(-qx));

    float dqx =  2.f*(
        dr10*qy + dr20*qz +
        dr01*qy + dr11*(-2.f*qx) + dr21*qw +
        dr02*qz + dr12*(-qw) + dr22*(-2.f*qx));

    float dqy =  2.f*(
        dr00*(-2.f*qy) + dr10*qx + dr20*(-qw) +
        dr01*qx +
        dr12*qz + dr21*qz + dr22*(-2.f*qy) +
        dr02*qw);

    float dqz =  2.f*(
        dr00*(-2.f*qz) + dr10*qw + dr20*qx +
        dr01*(-qw) + dr11*(-2.f*qz) +
        dr12*qy + dr21*qy +
        dr02*qx);

    // --- project through normalization Jacobian ---
    float dot = dqw*qw + dqx*qx + dqy*qy + dqz*qz;
    grad_rot_w[i] = norm_inv * (dqw - dot*qw);
    grad_rot_x[i] = norm_inv * (dqx - dot*qx);
    grad_rot_y[i] = norm_inv * (dqy - dot*qy);
    grad_rot_z[i] = norm_inv * (dqz - dot*qz);

    // --- color gradients: chain through linear SH ---
    grad_color_DC_SH_r[i] = grad_color_lin_r[i] * C0;
    grad_color_DC_SH_g[i] = grad_color_lin_g[i] * C0;
    grad_color_DC_SH_b[i] = grad_color_lin_b[i] * C0;

    // --- sigmoid backward: dL/dlogit = dL/dopacity * s * (1 - s) ---
    float s = opacity[i];
    grad_logit_opacity[i] = grad_opacity[i] * s * (1.f - s);
}

/* ===== ===== Lifecycle ===== ===== */

void GaussActivLayer::allocate(int count)
{
    allocatedCount = count;
    output.allocate(count);
    gradInput.allocate(count);
}

void GaussActivLayer::zero_grad()
{
    gradInput.zero_grad();
}

/* ===== ===== Forward / Backward ===== ===== */

void GaussActivLayer::forward()
{
    int count = input->count;
    output.count = count;

    size_t bytes = count * sizeof(float);

    // pass-throughs
    cudaMemcpy(output.pos_x,   input->pos_x,   bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.pos_y,   input->pos_y,   bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.pos_z,   input->pos_z,   bytes, cudaMemcpyDeviceToDevice);

    // color and opacity needs to be activated in forward
    // color is stored as DC SH coefficients, activated in kernel to linear RGB
    // opacity is stored as logit, activated in kernel with sigmoid

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    covForwardKernel<<<blocks, threads>>>(
        input->scale_x, input->scale_y, input->scale_z,
        input->rot_w,   input->rot_x,   input->rot_y,   input->rot_z,
        input->color_sh_r, input->color_sh_g, input->color_sh_b,
        input->opacity,   // logit-opacity in
        output.cov_xx, output.cov_xy, output.cov_xz,
        output.cov_yy, output.cov_yz, output.cov_zz,
        output.color_r, output.color_g, output.color_b,
        output.opacity,   // activated opacity out
        count
    );
    CUDA_SYNC_CHECK();
}

void GaussActivLayer::backward()
{
    int count = input->count;
    size_t bytes = count * sizeof(float);

    // position gradients pass through directly
    cudaMemcpy(gradInput.grad_pos_x, gradOutput->grad_pos_x, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gradInput.grad_pos_y, gradOutput->grad_pos_y, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(gradInput.grad_pos_z, gradOutput->grad_pos_z, bytes, cudaMemcpyDeviceToDevice);
    // opacity grad is NOT passed through — it flows back through sigmoid in the kernel below

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    covBackwardKernel<<<blocks, threads>>>(
        input->scale_x, input->scale_y, input->scale_z,
        input->rot_w,   input->rot_x,   input->rot_y,   input->rot_z,
        output.opacity,   // saved activated opacity for sigmoid backward
        gradOutput->grad_cov_xx,  gradOutput->grad_cov_xy,  gradOutput->grad_cov_xz,
        gradOutput->grad_cov_yy,  gradOutput->grad_cov_yz,  gradOutput->grad_cov_zz,
        gradOutput->grad_color_r, gradOutput->grad_color_g, gradOutput->grad_color_b,
        gradOutput->grad_opacity, // activated opacity
        gradInput.grad_scale_x, gradInput.grad_scale_y, gradInput.grad_scale_z,
        gradInput.grad_rot_w,   gradInput.grad_rot_x,
        gradInput.grad_rot_y,   gradInput.grad_rot_z,
        gradInput.grad_color_sh_r, gradInput.grad_color_sh_g, gradInput.grad_color_sh_b,
        gradInput.grad_opacity, // logit-opacity
        count
    );
    CUDA_SYNC_CHECK();
}
