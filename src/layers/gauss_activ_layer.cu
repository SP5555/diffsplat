#include "gauss_activ_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
#include "../utils/sh_consts.h"

/* ===== ===== Kernels ===== ===== */

/**
 * Forward kernel: computes 3D covariance Cov = M*M^T where M = R*S,
 * and evaluates spherical harmonic color for the given view direction.
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
 * SH color evaluation (degrees 0-3, Condon-Shortley convention):
 *   view dir = normalize(cam_pos - splat_pos)
 *   color = sum_over_bands(SH_coeff * Y(dir)) + 0.5, clamped to [0,1]
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
    // higher-order SH (may be nullptr if sh_degree == 0)
    // layout: sh_rest_r[band * count + i]
    const float *__restrict__ sh_rest_r,
    const float *__restrict__ sh_rest_g,
    const float *__restrict__ sh_rest_b,
    int sh_degree,
    // camera position for view direction
    float cam_x, float cam_y, float cam_z,
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

    // normalize quaternion (fmaxf guards against all-zero input -> rsqrtf(0) = inf)
    float qw = i_rw[i], qx = i_rx[i], qy = i_ry[i], qz = i_rz[i];
    float norm = rsqrtf(fmaxf(qw*qw + qx*qx + qy*qy + qz*qz, 1e-12f));
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

    // ===== SH color evaluation =====
    // view direction: splat -> camera, normalized (standard 3DGS convention)
    float dx = cam_x - i_x[i];
    float dy = cam_y - i_y[i];
    float dz = cam_z - i_z[i];
    float inv_len = rsqrtf(fmaxf(dx*dx + dy*dy + dz*dz, 1e-12f));
    float nx = dx * inv_len;
    float ny = dy * inv_len;
    float nz = dz * inv_len;

    float cr = i_DC_SH_R[i] * SH_C0;
    float cg = i_DC_SH_G[i] * SH_C0;
    float cb = i_DC_SH_B[i] * SH_C0;

    // branches are uniform across the warp (same sh_degree for all threads)
    if (sh_degree >= 1)
    {
        cr += SH_C1 * (-ny * sh_rest_r[0*count+i] + nz * sh_rest_r[1*count+i] - nx * sh_rest_r[2*count+i]);
        cg += SH_C1 * (-ny * sh_rest_g[0*count+i] + nz * sh_rest_g[1*count+i] - nx * sh_rest_g[2*count+i]);
        cb += SH_C1 * (-ny * sh_rest_b[0*count+i] + nz * sh_rest_b[1*count+i] - nx * sh_rest_b[2*count+i]);
    }
    if (sh_degree >= 2)
    {
        float xx = nx*nx, yy = ny*ny, zz = nz*nz;
        float xy = nx*ny, yz = ny*nz, xz = nx*nz;
        cr += SH_C2_0 * xy            * sh_rest_r[3*count+i]
            + SH_C2_1 * yz            * sh_rest_r[4*count+i]
            + SH_C2_2 * (2*zz-xx-yy) * sh_rest_r[5*count+i]
            + SH_C2_3 * xz            * sh_rest_r[6*count+i]
            + SH_C2_4 * (xx-yy)       * sh_rest_r[7*count+i];
        cg += SH_C2_0 * xy            * sh_rest_g[3*count+i]
            + SH_C2_1 * yz            * sh_rest_g[4*count+i]
            + SH_C2_2 * (2*zz-xx-yy) * sh_rest_g[5*count+i]
            + SH_C2_3 * xz            * sh_rest_g[6*count+i]
            + SH_C2_4 * (xx-yy)       * sh_rest_g[7*count+i];
        cb += SH_C2_0 * xy            * sh_rest_b[3*count+i]
            + SH_C2_1 * yz            * sh_rest_b[4*count+i]
            + SH_C2_2 * (2*zz-xx-yy) * sh_rest_b[5*count+i]
            + SH_C2_3 * xz            * sh_rest_b[6*count+i]
            + SH_C2_4 * (xx-yy)       * sh_rest_b[7*count+i];
    }
    if (sh_degree >= 3)
    {
        float xx = nx*nx, yy = ny*ny, zz = nz*nz;
        float xy = nx*ny;
        cr += SH_C3_0 * ny*(3*xx-yy)       * sh_rest_r[ 8*count+i]
            + SH_C3_1 * xy*nz               * sh_rest_r[ 9*count+i]
            + SH_C3_2 * ny*(4*zz-xx-yy)     * sh_rest_r[10*count+i]
            + SH_C3_3 * nz*(2*zz-3*xx-3*yy) * sh_rest_r[11*count+i]
            + SH_C3_4 * nx*(4*zz-xx-yy)     * sh_rest_r[12*count+i]
            + SH_C3_5 * nz*(xx-yy)           * sh_rest_r[13*count+i]
            + SH_C3_6 * nx*(xx-3*yy)         * sh_rest_r[14*count+i];
        cg += SH_C3_0 * ny*(3*xx-yy)       * sh_rest_g[ 8*count+i]
            + SH_C3_1 * xy*nz               * sh_rest_g[ 9*count+i]
            + SH_C3_2 * ny*(4*zz-xx-yy)     * sh_rest_g[10*count+i]
            + SH_C3_3 * nz*(2*zz-3*xx-3*yy) * sh_rest_g[11*count+i]
            + SH_C3_4 * nx*(4*zz-xx-yy)     * sh_rest_g[12*count+i]
            + SH_C3_5 * nz*(xx-yy)           * sh_rest_g[13*count+i]
            + SH_C3_6 * nx*(xx-3*yy)         * sh_rest_g[14*count+i];
        cb += SH_C3_0 * ny*(3*xx-yy)       * sh_rest_b[ 8*count+i]
            + SH_C3_1 * xy*nz               * sh_rest_b[ 9*count+i]
            + SH_C3_2 * ny*(4*zz-xx-yy)     * sh_rest_b[10*count+i]
            + SH_C3_3 * nz*(2*zz-3*xx-3*yy) * sh_rest_b[11*count+i]
            + SH_C3_4 * nx*(4*zz-xx-yy)     * sh_rest_b[12*count+i]
            + SH_C3_5 * nz*(xx-yy)           * sh_rest_b[13*count+i]
            + SH_C3_6 * nx*(xx-3*yy)         * sh_rest_b[14*count+i];
    }

    o_lin_R[i] = fminf(fmaxf(cr + 0.5f, 0.f), 1.f);
    o_lin_G[i] = fminf(fmaxf(cg + 0.5f, 0.f), 1.f);
    o_lin_B[i] = fminf(fmaxf(cb + 0.5f, 0.f), 1.f);

    // sigmoid activation on opacity
    o_A[i] = 1.f / (1.f + expf(-i_logit_A[i]));
}

/**
 * Backward kernel: chains dL/dCov back to log-scale and quaternion gradients,
 * dL/dcolor back through SH to SH coefficients,
 * and dL/dopacity back through sigmoid to logit-opacity.
 *
 * View direction is treated as constant (not differentiated w.r.t. splat position).
 * This is the standard 3DGS approximation.
 *
 * Gradient through clamp [0,1]: zero when output is at boundary (read from out.color).
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
    const float *__restrict__ pos_x,
    const float *__restrict__ pos_y,
    const float *__restrict__ pos_z,
    const float *__restrict__ opacity,
    // forward outputs (for clamp check)
    const float *__restrict__ fwd_lin_R,
    const float *__restrict__ fwd_lin_G,
    const float *__restrict__ fwd_lin_B,
    // higher-order SH (may be nullptr if sh_degree == 0)
    const float *__restrict__ sh_rest_r,
    const float *__restrict__ sh_rest_g,
    const float *__restrict__ sh_rest_b,
    int sh_degree,
    float cam_x, float cam_y, float cam_z,
    // gradient output (from next layer)
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
    // gradient input (to this layer's parameters)
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
    float *grad_i_sh_rest_r,
    float *grad_i_sh_rest_g,
    float *grad_i_sh_rest_b,
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
    float norm_inv = rsqrtf(fmaxf(qw_raw*qw_raw + qx_raw*qx_raw + qy_raw*qy_raw + qz_raw*qz_raw, 1e-12f));
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

    // --- SH color gradients ---
    // Gradient through clamp: zero if output was clamped (at 0 or 1 boundary)
    float dR = (fwd_lin_R[i] > 0.f && fwd_lin_R[i] < 1.f) ? grad_o_lin_r[i] : 0.f;
    float dG = (fwd_lin_G[i] > 0.f && fwd_lin_G[i] < 1.f) ? grad_o_lin_g[i] : 0.f;
    float dB = (fwd_lin_B[i] > 0.f && fwd_lin_B[i] < 1.f) ? grad_o_lin_b[i] : 0.f;

    // DC
    grad_i_DC_SH_r[i] = dR * SH_C0;
    grad_i_DC_SH_g[i] = dG * SH_C0;
    grad_i_DC_SH_b[i] = dB * SH_C0;

    if (sh_degree >= 1)
    {
        // recompute view direction
        float ddx = cam_x - pos_x[i], ddy = cam_y - pos_y[i], ddz = cam_z - pos_z[i];
        float il = rsqrtf(fmaxf(ddx*ddx + ddy*ddy + ddz*ddz, 1e-12f));
        float nx = ddx*il, ny = ddy*il, nz = ddz*il;

        // dL/dSH_lm = dL/dcolor * Y_lm(dir)
        // degree 1
        float y0 = SH_C1 * (-ny);
        float y1 = SH_C1 * ( nz);
        float y2 = SH_C1 * (-nx);
        grad_i_sh_rest_r[0*count+i] = dR * y0;
        grad_i_sh_rest_r[1*count+i] = dR * y1;
        grad_i_sh_rest_r[2*count+i] = dR * y2;
        grad_i_sh_rest_g[0*count+i] = dG * y0;
        grad_i_sh_rest_g[1*count+i] = dG * y1;
        grad_i_sh_rest_g[2*count+i] = dG * y2;
        grad_i_sh_rest_b[0*count+i] = dB * y0;
        grad_i_sh_rest_b[1*count+i] = dB * y1;
        grad_i_sh_rest_b[2*count+i] = dB * y2;

        if (sh_degree >= 2)
        {
            float xx = nx*nx, yy = ny*ny, zz = nz*nz;
            float xy = nx*ny, yz = ny*nz, xz = nx*nz;
            float b3 = SH_C2_0*xy,       b4 = SH_C2_1*yz;
            float b5 = SH_C2_2*(2*zz-xx-yy);
            float b6 = SH_C2_3*xz,       b7 = SH_C2_4*(xx-yy);
            grad_i_sh_rest_r[3*count+i] = dR*b3; grad_i_sh_rest_r[4*count+i] = dR*b4;
            grad_i_sh_rest_r[5*count+i] = dR*b5; grad_i_sh_rest_r[6*count+i] = dR*b6;
            grad_i_sh_rest_r[7*count+i] = dR*b7;
            grad_i_sh_rest_g[3*count+i] = dG*b3; grad_i_sh_rest_g[4*count+i] = dG*b4;
            grad_i_sh_rest_g[5*count+i] = dG*b5; grad_i_sh_rest_g[6*count+i] = dG*b6;
            grad_i_sh_rest_g[7*count+i] = dG*b7;
            grad_i_sh_rest_b[3*count+i] = dB*b3; grad_i_sh_rest_b[4*count+i] = dB*b4;
            grad_i_sh_rest_b[5*count+i] = dB*b5; grad_i_sh_rest_b[6*count+i] = dB*b6;
            grad_i_sh_rest_b[7*count+i] = dB*b7;

            if (sh_degree >= 3)
            {
                float c8  = SH_C3_0*ny*(3*xx-yy);
                float c9  = SH_C3_1*xy*nz;
                float c10 = SH_C3_2*ny*(4*zz-xx-yy);
                float c11 = SH_C3_3*nz*(2*zz-3*xx-3*yy);
                float c12 = SH_C3_4*nx*(4*zz-xx-yy);
                float c13 = SH_C3_5*nz*(xx-yy);
                float c14 = SH_C3_6*nx*(xx-3*yy);
                grad_i_sh_rest_r[ 8*count+i] = dR*c8;  grad_i_sh_rest_r[ 9*count+i] = dR*c9;
                grad_i_sh_rest_r[10*count+i] = dR*c10; grad_i_sh_rest_r[11*count+i] = dR*c11;
                grad_i_sh_rest_r[12*count+i] = dR*c12; grad_i_sh_rest_r[13*count+i] = dR*c13;
                grad_i_sh_rest_r[14*count+i] = dR*c14;
                grad_i_sh_rest_g[ 8*count+i] = dG*c8;  grad_i_sh_rest_g[ 9*count+i] = dG*c9;
                grad_i_sh_rest_g[10*count+i] = dG*c10; grad_i_sh_rest_g[11*count+i] = dG*c11;
                grad_i_sh_rest_g[12*count+i] = dG*c12; grad_i_sh_rest_g[13*count+i] = dG*c13;
                grad_i_sh_rest_g[14*count+i] = dG*c14;
                grad_i_sh_rest_b[ 8*count+i] = dB*c8;  grad_i_sh_rest_b[ 9*count+i] = dB*c9;
                grad_i_sh_rest_b[10*count+i] = dB*c10; grad_i_sh_rest_b[11*count+i] = dB*c11;
                grad_i_sh_rest_b[12*count+i] = dB*c12; grad_i_sh_rest_b[13*count+i] = dB*c13;
                grad_i_sh_rest_b[14*count+i] = dB*c14;
            }
        }
    }

    // --- sigmoid backward: dL/dlogit = dL/dopacity * s * (1 - s) ---
    float s = opacity[i];
    grad_i_logit_A[i] = grad_o_A[i] * s * (1.f - s);
}

/* ===== ===== Lifecycle ===== ===== */

void GaussActivLayer::allocate(int count)
{
    out.allocate(count);
}

void GaussActivLayer::allocateGrad(int count)
{
    grad_in.allocate(count, sh_degree);
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

    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    covForwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->pos_x,   in->pos_y,   in->pos_z,
        in->scale_x, in->scale_y, in->scale_z,
        in->rot_w,   in->rot_x,   in->rot_y,   in->rot_z,
        in->color_sh_r,
        in->color_sh_g,
        in->color_sh_b,
        in->logit_opacity,
        in->sh_num_bands > 0 ? (const float *)in->sh_rest_r : nullptr,
        in->sh_num_bands > 0 ? (const float *)in->sh_rest_g : nullptr,
        in->sh_num_bands > 0 ? (const float *)in->sh_rest_b : nullptr,
        sh_degree,
        cam_x, cam_y, cam_z,
        out.pos_x,  out.pos_y,  out.pos_z,
        out.cov_xx, out.cov_xy, out.cov_xz,
        out.cov_yy, out.cov_yz, out.cov_zz,
        out.color_r,
        out.color_g,
        out.color_b,
        out.opacity,
        count
    );
    CUDA_SYNC_CHECK();
}

void GaussActivLayer::backward()
{
    int count = in->count;

    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    covBackwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->scale_x, in->scale_y, in->scale_z,
        in->rot_w,   in->rot_x,   in->rot_y,   in->rot_z,
        in->pos_x,   in->pos_y,   in->pos_z,
        out.opacity,
        out.color_r, out.color_g, out.color_b,   // saved forward outputs for clamp check
        in->sh_num_bands > 0 ? (const float *)in->sh_rest_r : nullptr,
        in->sh_num_bands > 0 ? (const float *)in->sh_rest_g : nullptr,
        in->sh_num_bands > 0 ? (const float *)in->sh_rest_b : nullptr,
        sh_degree,
        cam_x, cam_y, cam_z,
        grad_out->grad_pos_x,  grad_out->grad_pos_y,  grad_out->grad_pos_z,
        grad_out->grad_cov_xx, grad_out->grad_cov_xy, grad_out->grad_cov_xz,
        grad_out->grad_cov_yy, grad_out->grad_cov_yz, grad_out->grad_cov_zz,
        grad_out->grad_color_r,
        grad_out->grad_color_g,
        grad_out->grad_color_b,
        grad_out->grad_opacity,
        grad_in.grad_pos_x,   grad_in.grad_pos_y,   grad_in.grad_pos_z,
        grad_in.grad_scale_x, grad_in.grad_scale_y, grad_in.grad_scale_z,
        grad_in.grad_rot_w,   grad_in.grad_rot_x,
        grad_in.grad_rot_y,   grad_in.grad_rot_z,
        grad_in.grad_color_sh_r,
        grad_in.grad_color_sh_g,
        grad_in.grad_color_sh_b,
        grad_in.grad_logit_opacity,
        grad_in.sh_num_bands > 0 ? (float *)grad_in.grad_sh_rest_r : nullptr,
        grad_in.sh_num_bands > 0 ? (float *)grad_in.grad_sh_rest_g : nullptr,
        grad_in.sh_num_bands > 0 ? (float *)grad_in.grad_sh_rest_b : nullptr,
        count
    );
    CUDA_SYNC_CHECK();
}
