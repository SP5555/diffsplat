#include "persp_project_layer.h"
#include <cuda_runtime.h>
#include <float.h>
#include <glm/gtc/type_ptr.hpp>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

__constant__ float d_pv[16]; // device copy of PV matrix (column-major, GLM layout)

/* ===== ===== Kernels ===== ===== */

/**
 * Forward kernel: projects Splat3D in world space to Splat2D in NDC space.
 *
 * PV matrix is column-major (GLM layout), indexed as pv[col*4 + row].
 *
 * Position:
 *   clip    = PV * [x y z 1]^T
 *   ndc.xyz = clip.xyz / clip.w
 *
 * Covariance (2x3 Jacobian of NDC w.r.t. world pos):
 *   JR0[k] = (pv[k*4+0] * c_w - pv[3*4+0] * c_x) / c_w^2   (d(ndc_x)/d[x,y,z])
 *   JR1[k] = (pv[k*4+1] * c_w - pv[3*4+1] * c_y) / c_w^2   (d(ndc_y)/d[x,y,z])
 *   Cov_2D = J * Cov_3D * J^T
 *
 * Culled splats (c_w <= 0) are flagged with pos_z = FLT_MAX for the rasterizer to skip.
 *
 * One thread is launched per splat.
 */
__global__ void perspProjectForwardKernel(
    // inputs (world space)
    const float *__restrict__ i_w_x,
    const float *__restrict__ i_w_y,
    const float *__restrict__ i_w_z,
    const float *__restrict__ i_w_cxx,
    const float *__restrict__ i_w_cxy,
    const float *__restrict__ i_w_cxz,
    const float *__restrict__ i_w_cyy,
    const float *__restrict__ i_w_cyz,
    const float *__restrict__ i_w_czz,
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
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = i_w_x[i], y = i_w_y[i], z = i_w_z[i];

    /*
        PV is column major
        PV = [ pv[0]  pv[4]  pv[8]  pv[12] ]
             [ pv[1]  pv[5]  pv[9]  pv[13] ]
             [ pv[2]  pv[6]  pv[10] pv[14] ]
             [ pv[3]  pv[7]  pv[11] pv[15] ]

        clip_pos = PV * [x y z 1]^T
                 = [ c_x  c_y  c_z  c_w ]^T
    */
    float c_x = d_pv[0]*x + d_pv[4]*y + d_pv[8]*z  + d_pv[12];
    float c_y = d_pv[1]*x + d_pv[5]*y + d_pv[9]*z  + d_pv[13];
    float c_z = d_pv[2]*x + d_pv[6]*y + d_pv[10]*z + d_pv[14];
    float c_w = d_pv[3]*x + d_pv[7]*y + d_pv[11]*z + d_pv[15];

    // cull splats behind camera
    if (c_w <= 0.f)
    {
        o_ndc_z[i] = FLT_MAX; // rasterizer will skip this
        return;
    }
    float inv_w  = 1.f / c_w;
    float inv_w2 = inv_w * inv_w;

    /*
        NDC position (perspective divide)

        ndc_pos = [ c_x/c_w  c_y/c_w  c_z/c_w ]^T
    */
    o_ndc_x[i] = c_x * inv_w;
    o_ndc_y[i] = c_y * inv_w;
    o_ndc_z[i] = c_z * inv_w;

    /*
        Jacobian of perspective projection (d(ndc)/d(world)) is 2x3:

        J = [ d(ndc_x)/dx  d(ndc_x)/dy  d(ndc_x)/dz ]
            [ d(ndc_y)/dx  d(ndc_y)/dy  d(ndc_y)/dz ]
          = [ d(c_x/c_w)/dx  d(c_x/c_w)/dy  d(c_x/c_w)/dz ]
            [ d(c_y/c_w)/dx  d(c_y/c_w)/dy  d(c_y/c_w)/dz ]
          = [ (pv[0]*c_w-cx*pv[3])/c_w^2  (pv[4]*c_w-cx*pv[7])/c_w^2  (pv[8]*c_w-cx*pv[11])/c_w^2 ]
            [ (pv[1]*c_w-cy*pv[3])/c_w^2  (pv[5]*c_w-cy*pv[7])/c_w^2  (pv[9]*c_w-cy*pv[11])/c_w^2 ]
            
    */
    float j00 = (d_pv[0] * c_w - c_x * d_pv[ 3]) * inv_w2;
    float j01 = (d_pv[4] * c_w - c_x * d_pv[ 7]) * inv_w2;
    float j02 = (d_pv[8] * c_w - c_x * d_pv[11]) * inv_w2;

    float j10 = (d_pv[1] * c_w - c_y * d_pv[ 3]) * inv_w2;
    float j11 = (d_pv[5] * c_w - c_y * d_pv[ 7]) * inv_w2;
    float j12 = (d_pv[9] * c_w - c_y * d_pv[11]) * inv_w2;

    // 3D covariance (symmetric)
    float sxx = i_w_cxx[i], sxy = i_w_cxy[i], sxz = i_w_cxz[i];
    float                   syy = i_w_cyy[i], syz = i_w_cyz[i];
    float                                     szz = i_w_czz[i];

    // J * Cov3D
    float js00 = j00*sxx + j01*sxy + j02*sxz;
    float js01 = j00*sxy + j01*syy + j02*syz;
    float js02 = j00*sxz + j01*syz + j02*szz;

    float js10 = j10*sxx + j11*sxy + j12*sxz;
    float js11 = j10*sxy + j11*syy + j12*syz;
    float js12 = j10*sxz + j11*syz + j12*szz;

    // Cov2D = J * Cov3D * J^T
    o_ndc_cxx[i] = j00*js00 + j01*js01 + j02*js02;
    o_ndc_cxy[i] = j00*js10 + j01*js11 + j02*js12;
    o_ndc_cyy[i] = j10*js10 + j11*js11 + j12*js12;

    // pass-throughs
    o_R[i] = i_R[i];
    o_G[i] = i_G[i];
    o_B[i] = i_B[i];
    o_A[i] = i_A[i];
}

/**
 * Backward kernel: chains dL/dSplat2D -> dL/dSplat3D.
 *
 * Position backward: dL/d(world) = J^T * dL/d(ndc)
 * Covariance backward: dL/dCov_3D = J^T * dL/dCov_2D_sym * J
 *
 * Camera matrices are treated as constants (no grad w.r.t. camera).
 *
 * One thread per splat.
 */
__global__ void perspProjectBackwardKernel(
    // splat3d inputs (recomputed, not saved)
    const float *__restrict__ w_x,
    const float *__restrict__ w_y,
    const float *__restrict__ w_z,
    const float *__restrict__ w_cxx,
    const float *__restrict__ w_cxy,
    const float *__restrict__ w_cxz,
    const float *__restrict__ w_cyy,
    const float *__restrict__ w_cyz,
    const float *__restrict__ w_czz,
    // gradient output
    const float *__restrict__ grad_o_ndc_x,
    const float *__restrict__ grad_o_ndc_y,
    const float *__restrict__ grad_o_ndc_cxx,
    const float *__restrict__ grad_o_ndc_cxy,
    const float *__restrict__ grad_o_ndc_cyy,
    const float *__restrict__ grad_o_R,
    const float *__restrict__ grad_o_G,
    const float *__restrict__ grad_o_B,
    const float *__restrict__ grad_o_A,
    // gradient input
    float *grad_i_w_x,
    float *grad_i_w_y,
    float *grad_i_w_z,
    float *grad_i_w_cxx,
    float *grad_i_w_cxy,
    float *grad_i_w_cxz,
    float *grad_i_w_cyy,
    float *grad_i_w_cyz,
    float *grad_i_w_czz,
    float *grad_i_R,
    float *grad_i_G,
    float *grad_i_B,
    float *grad_i_A,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = w_x[i], y = w_y[i], z = w_z[i];

    // recompute clip coords
    float c_x = d_pv[0]*x + d_pv[4]*y + d_pv[8]*z  + d_pv[12];
    float c_y = d_pv[1]*x + d_pv[5]*y + d_pv[9]*z  + d_pv[13];
    float c_w = d_pv[3]*x + d_pv[7]*y + d_pv[11]*z + d_pv[15];

    // skip culled splats
    if (c_w <= 0.f)
    {
        grad_i_w_x[i] = 0.f;   grad_i_w_y[i] = 0.f;   grad_i_w_z[i] = 0.f;
        grad_i_w_cxx[i] = 0.f; grad_i_w_cxy[i] = 0.f; grad_i_w_cxz[i] = 0.f;
        grad_i_w_cyy[i] = 0.f; grad_i_w_cyz[i] = 0.f; grad_i_w_czz[i] = 0.f;
        grad_i_R[i] = 0.f;     grad_i_G[i] = 0.f;     grad_i_B[i] = 0.f;
        grad_i_A[i] = 0.f;
        return;
    }

    float inv_w  = 1.f / c_w;
    float inv_w2 = inv_w * inv_w;

    // recompute Jacobian
    float j00 = (d_pv[0] * c_w - c_x * d_pv[ 3]) * inv_w2;
    float j01 = (d_pv[4] * c_w - c_x * d_pv[ 7]) * inv_w2;
    float j02 = (d_pv[8] * c_w - c_x * d_pv[11]) * inv_w2;

    float j10 = (d_pv[1] * c_w - c_y * d_pv[ 3]) * inv_w2;
    float j11 = (d_pv[5] * c_w - c_y * d_pv[ 7]) * inv_w2;
    float j12 = (d_pv[9] * c_w - c_y * d_pv[11]) * inv_w2;

    float sxx = w_cxx[i], sxy = w_cxy[i], sxz = w_cxz[i];
    float                 syy = w_cyy[i], syz = w_cyz[i];
    float                                 szz = w_czz[i];

    // J * Cov3D
    float js00 = j00*sxx + j01*sxy + j02*sxz;
    float js01 = j00*sxy + j01*syy + j02*syz;
    float js02 = j00*sxz + j01*syz + j02*szz;

    float js10 = j10*sxx + j11*sxy + j12*sxz;
    float js11 = j10*sxy + j11*syy + j12*syz;
    float js12 = j10*sxz + j11*syz + j12*szz;

    float sxx_2D = grad_o_ndc_cxx[i];
    float sxy_2D = grad_o_ndc_cxy[i];
    float syy_2D = grad_o_ndc_cyy[i];

    // dL/dCov_3D = 2*d_cxx*(JR0 outer s_jr0)
    //            + d_cxy*(JR0 outer s_jr1 + JR1 outer s_jr0)
    //            + 2*d_cyy*(JR1 outer s_jr1)
    grad_i_w_cxx[i] = 2.f*sxx_2D*j00*js00 + sxy_2D*(j00*js10 + j10*js00) + 2.f*syy_2D*j10*js10;
    grad_i_w_cxy[i] = 2.f*sxx_2D*j00*js01 + sxy_2D*(j00*js11 + j10*js01) + 2.f*syy_2D*j10*js11;
    grad_i_w_cxz[i] = 2.f*sxx_2D*j00*js02 + sxy_2D*(j00*js12 + j10*js02) + 2.f*syy_2D*j10*js12;
    grad_i_w_cyy[i] = 2.f*sxx_2D*j01*js01 + sxy_2D*(j01*js11 + j11*js01) + 2.f*syy_2D*j11*js11;
    grad_i_w_cyz[i] = 2.f*sxx_2D*j01*js02 + sxy_2D*(j01*js12 + j11*js02) + 2.f*syy_2D*j11*js12;
    grad_i_w_czz[i] = 2.f*sxx_2D*j02*js02 + sxy_2D*(j02*js12 + j12*js02) + 2.f*syy_2D*j12*js12;

    // ===== position backward =====
    // ndc = clip / c_w, so d(ndc)/d(world) = J (already computed above)
    // dL/d(world) = J^T * dL/d(ndc)
    float ndc_x = grad_o_ndc_x[i];
    float ndc_y = grad_o_ndc_y[i];

    grad_i_w_x[i] = ndc_x * j00 + ndc_y * j10;
    grad_i_w_y[i] = ndc_x * j01 + ndc_y * j11;
    grad_i_w_z[i] = ndc_x * j02 + ndc_y * j12;

    // pass-through gradients
    grad_i_R[i] = grad_o_R[i];
    grad_i_G[i] = grad_o_G[i];
    grad_i_B[i] = grad_o_B[i];
    grad_i_A[i] = grad_o_A[i];
}

/* ===== ===== Lifecycle ===== ===== */

void PerspProjectLayer::allocate(int count)
{
    allocated_count = count;
    out.allocate(count);
    grad_in.allocate(count);
}

void PerspProjectLayer::zero_grad()
{
    grad_in.zero_grad();
}

void PerspProjectLayer::setCamera(const glm::mat4 &view, const glm::mat4 &proj)
{
    glm::mat4 pv = proj * view;
    memcpy(h_pv, glm::value_ptr(pv), 16 * sizeof(float));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pv, h_pv, 16 * sizeof(float)));
}

/* ===== ===== Forward / Backward ===== ===== */

void PerspProjectLayer::forward()
{
    int count   = in->count;
    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    perspProjectForwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->pos_x,   in->pos_y,   in->pos_z,
        in->cov_xx,  in->cov_xy,  in->cov_xz,
        in->cov_yy,  in->cov_yz,  in->cov_zz,
        in->color_r, in->color_g, in->color_b,
        in->opacity,
        out.pos_x,   out.pos_y,   out.pos_z,
        out.cov_xx,  out.cov_xy,  out.cov_yy,
        out.color_r, out.color_g, out.color_b,
        out.opacity,
        count
    );
    CUDA_SYNC_CHECK();
}

void PerspProjectLayer::backward()
{
    int count   = in->count;
    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    perspProjectBackwardKernel<<<blocks, BLOCK_SIZE>>>(
        in->pos_x,  in->pos_y,  in->pos_z,
        in->cov_xx, in->cov_xy, in->cov_xz,
        in->cov_yy, in->cov_yz, in->cov_zz,
        grad_out->grad_pos_x,   grad_out->grad_pos_y,
        grad_out->grad_cov_xx,  grad_out->grad_cov_xy,  grad_out->grad_cov_yy,
        grad_out->grad_color_r, grad_out->grad_color_g, grad_out->grad_color_b,
        grad_out->grad_opacity,
        grad_in.grad_pos_x,   grad_in.grad_pos_y,   grad_in.grad_pos_z,
        grad_in.grad_cov_xx,  grad_in.grad_cov_xy,  grad_in.grad_cov_xz,
        grad_in.grad_cov_yy,  grad_in.grad_cov_yz,  grad_in.grad_cov_zz,
        grad_in.grad_color_r, grad_in.grad_color_g, grad_in.grad_color_b,
        grad_in.grad_opacity,
        count
    );
    CUDA_SYNC_CHECK();
}
