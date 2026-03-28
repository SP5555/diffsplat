#include "gauss_activ_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
static constexpr float C0 = 0.282095f;  // DC SH coefficient

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
    // inputs: pos, log-scale, raw quaternion,
    //         SH coefficient RGB, logit-opacity
    const float *__restrict__ i_x,
    const float *__restrict__ i_y,
    const float *__restrict__ i_z,
    const float *__restrict__ i_sx,
    const float *__restrict__ i_sy,
    const float *__restrict__ i_sz,
    const float *__restrict__ i_rw,
    const float *__restrict__ i_rx,
    const float *__restrict__ i_ry,
    const float *__restrict__ i_rz,
    const float *__restrict__ i_DC_SH_R,
    const float *__restrict__ i_DC_SH_G,
    const float *__restrict__ i_DC_SH_B,
    const float *__restrict__ i_logit_A,
    // outputs: 3D covariance + RGBA
    float *o_x,   float *o_y,   float *o_z,
    float *o_cxx, float *o_cxy, float *o_cxz,
    float *o_cyy, float *o_cyz, float *o_czz,
    float *o_lin_R,
    float *o_lin_G,
    float *o_lin_B,
    float *o_A,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // pass-through position
    o_x[i] = i_x[i];
    o_y[i] = i_y[i];
    o_z[i] = i_z[i];

    // normalize quaternion
    float qw = i_rw[i], qx = i_rx[i], qy = i_ry[i], qz = i_rz[i];
    float norm = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    qw *= norm; qx *= norm; qy *= norm; qz *= norm;

    // actual scale
    float sx = expf(i_sx[i]);
    float sy = expf(i_sy[i]);
    float sz = expf(i_sz[i]);

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
    o_cxx[i] = m00*m00 + m01*m01 + m02*m02;
    o_cxy[i] = m00*m10 + m01*m11 + m02*m12;
    o_cxz[i] = m00*m20 + m01*m21 + m02*m22;
    o_cyy[i] = m10*m10 + m11*m11 + m12*m12;
    o_cyz[i] = m10*m20 + m11*m21 + m12*m22;
    o_czz[i] = m20*m20 + m21*m21 + m22*m22;
    // for fun!
    // o_cxx[i] = 1.f;
    // o_cxy[i] = 0.f;
    // o_cxz[i] = 0.f;
    // o_cyy[i] = 1.f;
    // o_cyz[i] = 0.f;
    // o_czz[i] = 1.f;

    // DC SH coefficient -> linear RGB
    o_lin_R[i] = fminf(fmaxf(i_DC_SH_R[i] * C0 + 0.5f, 0.f), 1.f);
    o_lin_G[i] = fminf(fmaxf(i_DC_SH_G[i] * C0 + 0.5f, 0.f), 1.f);
    o_lin_B[i] = fminf(fmaxf(i_DC_SH_B[i] * C0 + 0.5f, 0.f), 1.f);

    // sigmoid activation on opacity
    o_A[i] = 1.f / (1.f + expf(-i_logit_A[i]));
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
    // inputs: log-scale and raw quaternion
    const float *__restrict__ scale_x,
    const float *__restrict__ scale_y,
    const float *__restrict__ scale_z,
    const float *__restrict__ rot_w,
    const float *__restrict__ rot_x,
    const float *__restrict__ rot_y,
    const float *__restrict__ rot_z,
    const float *__restrict__ opacity,
    // gradient output
    const float *__restrict__ grad_o_x,
    const float *__restrict__ grad_o_y,
    const float *__restrict__ grad_o_z,
    const float *__restrict__ grad_o_cxx,
    const float *__restrict__ grad_o_cxy,
    const float *__restrict__ grad_o_cxz,
    const float *__restrict__ grad_o_cyy,
    const float *__restrict__ grad_o_cyz,
    const float *__restrict__ grad_o_czz,
    const float *__restrict__ grad_o_lin_r,
    const float *__restrict__ grad_o_lin_g,
    const float *__restrict__ grad_o_lin_b,
    const float *__restrict__ grad_o_A,
    // gradient input
    float *grad_i_x,
    float *grad_i_y,
    float *grad_i_z,
    float *grad_i_sx,
    float *grad_i_sy,
    float *grad_i_sz,
    float *grad_i_rw,
    float *grad_i_rx,
    float *grad_i_ry,
    float *grad_i_rz,
    float *grad_i_DC_SH_r,
    float *grad_i_DC_SH_g,
    float *grad_i_DC_SH_b,
    float *grad_i_logit_A,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // pass-through position gradients
    grad_i_x[i] = grad_o_x[i];
    grad_i_y[i] = grad_o_y[i];
    grad_i_z[i] = grad_o_z[i];

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
    float dxx = grad_o_cxx[i];
    float dxy = grad_o_cxy[i];
    float dxz = grad_o_cxz[i];
    float dyy = grad_o_cyy[i];
    float dyz = grad_o_cyz[i];
    float dzz = grad_o_czz[i];

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
    grad_i_sx[i] = (dm00*r00 + dm10*r10 + dm20*r20) * sx;
    grad_i_sy[i] = (dm01*r01 + dm11*r11 + dm21*r21) * sy;
    grad_i_sz[i] = (dm02*r02 + dm12*r12 + dm22*r22) * sz;

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
    grad_i_rw[i] = norm_inv * (dqw - dot*qw);
    grad_i_rx[i] = norm_inv * (dqx - dot*qx);
    grad_i_ry[i] = norm_inv * (dqy - dot*qy);
    grad_i_rz[i] = norm_inv * (dqz - dot*qz);

    // --- color gradients: chain through linear SH ---
    grad_i_DC_SH_r[i] = grad_o_lin_r[i] * C0;
    grad_i_DC_SH_g[i] = grad_o_lin_g[i] * C0;
    grad_i_DC_SH_b[i] = grad_o_lin_b[i] * C0;

    // --- sigmoid backward: dL/dlogit = dL/dopacity * s * (1 - s) ---
    float s = opacity[i];
    grad_i_logit_A[i] = grad_o_A[i] * s * (1.f - s);
}

/* ===== ===== Lifecycle ===== ===== */

void GaussActivLayer::allocate(int count)
{
    allocated_count = count;
    out.allocate(count);
    grad_in.allocate(count);
}

void GaussActivLayer::zero_grad()
{
    grad_in.zero_grad();
}

/* ===== ===== Forward / Backward ===== ===== */

void GaussActivLayer::forward()
{
    int count = in->count;
    out.count = count;

    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    covForwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->pos_x,   in->pos_y,   in->pos_z,
        in->scale_x, in->scale_y, in->scale_z,
        in->rot_w,   in->rot_x,   in->rot_y,   in->rot_z,
        in->color_sh_r,
        in->color_sh_g,
        in->color_sh_b,
        in->logit_opacity,   // logit-opacity in
        out.pos_x,  out.pos_y,  out.pos_z,
        out.cov_xx, out.cov_xy, out.cov_xz,
        out.cov_yy, out.cov_yz, out.cov_zz,
        out.color_r,
        out.color_g,
        out.color_b,
        out.opacity,   // activated opacity out
        count
    );
    CUDA_SYNC_CHECK();
}

void GaussActivLayer::backward()
{
    int count = in->count;

    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    covBackwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->scale_x, in->scale_y, in->scale_z,
        in->rot_w,   in->rot_x,   in->rot_y,   in->rot_z,
        out.opacity,   // saved activated opacity for sigmoid backward
        grad_out->grad_pos_x,  grad_out->grad_pos_y,  grad_out->grad_pos_z,
        grad_out->grad_cov_xx, grad_out->grad_cov_xy, grad_out->grad_cov_xz,
        grad_out->grad_cov_yy, grad_out->grad_cov_yz, grad_out->grad_cov_zz,
        grad_out->grad_color_r,
        grad_out->grad_color_g,
        grad_out->grad_color_b,
        grad_out->grad_opacity, // activated opacity
        grad_in.grad_pos_x,   grad_in.grad_pos_y,   grad_in.grad_pos_z,
        grad_in.grad_scale_x, grad_in.grad_scale_y, grad_in.grad_scale_z,
        grad_in.grad_rot_w,   grad_in.grad_rot_x,
        grad_in.grad_rot_y,   grad_in.grad_rot_z,
        grad_in.grad_color_sh_r,
        grad_in.grad_color_sh_g,
        grad_in.grad_color_sh_b,
        grad_in.grad_logit_opacity, // logit-opacity
        count
    );
    CUDA_SYNC_CHECK();
}
