#include "persp_project_layer.h"
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>

#include "../utils/cuda_utils.cuh"

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
 * Covariance (2x3 Jacobian of NDC w.r.t. world pos, quotient rule on clip/c_w):
 *   JR0[k] = (pv[k*4+0] * c_w - pv[3*4+0] * c_x) / c_w^2   (d(ndc_x)/d[x,y,z])
 *   JR1[k] = (pv[k*4+1] * c_w - pv[3*4+1] * c_y) / c_w^2   (d(ndc_y)/d[x,y,z])
 *   Cov_2D = J * Cov_3D * J^T
 *
 * Culled splats (c_w <= 0) are flagged with pos_z = FLT_MAX for the rasterizer to skip.
 *
 * One thread is launched per splat.
 */
__global__ void perspProjectForwardKernel(
    // splat3d inputs (world space)
    const float *__restrict__ pos_x,
    const float *__restrict__ pos_y,
    const float *__restrict__ pos_z,
    const float *__restrict__ cov_xx,
    const float *__restrict__ cov_xy,
    const float *__restrict__ cov_xz,
    const float *__restrict__ cov_yy,
    const float *__restrict__ cov_yz,
    const float *__restrict__ cov_zz,
    const float *__restrict__ color_r,
    const float *__restrict__ color_g,
    const float *__restrict__ color_b,
    const float *__restrict__ opacity,
    // camera
    const float *__restrict__ pv, // [16] column-major PV = P * V
    // splat2d outputs (NDC space)
    float *out_pos_x,
    float *out_pos_y,
    float *out_pos_z,
    float *out_cov_xx,
    float *out_cov_xy,
    float *out_cov_yy,
    float *out_color_r,
    float *out_color_g,
    float *out_color_b,
    float *out_opacity,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = pos_x[i], y = pos_y[i], z = pos_z[i];

    // clip = PV * [x y z 1]^T  (column-major: pv[col*4 + row])
    float c_x = pv[0]*x + pv[4]*y + pv[8]*z  + pv[12];
    float c_y = pv[1]*x + pv[5]*y + pv[9]*z  + pv[13];
    float c_z = pv[2]*x + pv[6]*y + pv[10]*z + pv[14];
    float c_w = pv[3]*x + pv[7]*y + pv[11]*z + pv[15];

    // cull splats behind camera
    if (c_w <= 0.f)
    {
        out_pos_z[i] = 3.402823466e+38f; // rasterizer will skip this
        return;
    }

    float inv_w  = 1.f / c_w;
    float inv_w2 = inv_w * inv_w;

    // NDC position
    out_pos_x[i] = c_x * inv_w;
    out_pos_y[i] = c_y * inv_w;
    out_pos_z[i] = c_z * inv_w;

    // Jacobian rows: d(ndc_x)/d[x,y,z] and d(ndc_y)/d[x,y,z]
    // pv column-major: row 0 entries are pv[0], pv[4], pv[8], pv[12]
    //                  row 1 entries are pv[1], pv[5], pv[9], pv[13]
    float jr0_x = (pv[0] * c_w - pv[ 3] * c_x) * inv_w2;
    float jr0_y = (pv[4] * c_w - pv[ 7] * c_x) * inv_w2;
    float jr0_z = (pv[8] * c_w - pv[11] * c_x) * inv_w2;

    float jr1_x = (pv[1] * c_w - pv[ 3] * c_y) * inv_w2;
    float jr1_y = (pv[5] * c_w - pv[ 7] * c_y) * inv_w2;
    float jr1_z = (pv[9] * c_w - pv[11] * c_y) * inv_w2;

    // 3D covariance (symmetric)
    float sxx = cov_xx[i], sxy = cov_xy[i], sxz = cov_xz[i];
    float                  syy = cov_yy[i], syz = cov_yz[i];
    float                                   szz = cov_zz[i];

    // Cov_3D * JR0 and Cov_3D * JR1
    float s_jr0_x = sxx*jr0_x + sxy*jr0_y + sxz*jr0_z;
    float s_jr0_y = sxy*jr0_x + syy*jr0_y + syz*jr0_z;
    float s_jr0_z = sxz*jr0_x + syz*jr0_y + szz*jr0_z;

    float s_jr1_x = sxx*jr1_x + sxy*jr1_y + sxz*jr1_z;
    float s_jr1_y = sxy*jr1_x + syy*jr1_y + syz*jr1_z;
    float s_jr1_z = sxz*jr1_x + syz*jr1_y + szz*jr1_z;

    // Cov_2D = J * Cov_3D * J^T
    out_cov_xx[i] = jr0_x*s_jr0_x + jr0_y*s_jr0_y + jr0_z*s_jr0_z;
    out_cov_xy[i] = jr0_x*s_jr1_x + jr0_y*s_jr1_y + jr0_z*s_jr1_z;
    out_cov_yy[i] = jr1_x*s_jr1_x + jr1_y*s_jr1_y + jr1_z*s_jr1_z;

    // pass-throughs
    out_color_r[i] = color_r[i];
    out_color_g[i] = color_g[i];
    out_color_b[i] = color_b[i];
    out_opacity[i] = opacity[i];
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
    const float *__restrict__ pos_x,
    const float *__restrict__ pos_y,
    const float *__restrict__ pos_z,
    const float *__restrict__ cov_xx,
    const float *__restrict__ cov_xy,
    const float *__restrict__ cov_xz,
    const float *__restrict__ cov_yy,
    const float *__restrict__ cov_yz,
    const float *__restrict__ cov_zz,
    // camera
    const float *__restrict__ pv,
    // grad output (from rasterizer, in NDC space)
    const float *__restrict__ grad_pos_x,
    const float *__restrict__ grad_pos_y,
    const float *__restrict__ grad_cov_xx,
    const float *__restrict__ grad_cov_xy,
    const float *__restrict__ grad_cov_yy,
    const float *__restrict__ grad_color_r,
    const float *__restrict__ grad_color_g,
    const float *__restrict__ grad_color_b,
    const float *__restrict__ grad_opacity,
    // grad input (to GaussActivLayer, in world space)
    float *grad_in_pos_x,
    float *grad_in_pos_y,
    float *grad_in_pos_z,
    float *grad_in_cov_xx,
    float *grad_in_cov_xy,
    float *grad_in_cov_xz,
    float *grad_in_cov_yy,
    float *grad_in_cov_yz,
    float *grad_in_cov_zz,
    float *grad_in_color_r,
    float *grad_in_color_g,
    float *grad_in_color_b,
    float *grad_in_opacity,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = pos_x[i], y = pos_y[i], z = pos_z[i];

    // recompute clip coords
    float c_x = pv[0]*x + pv[4]*y + pv[8]*z  + pv[12];
    float c_y = pv[1]*x + pv[5]*y + pv[9]*z  + pv[13];
    float c_w = pv[3]*x + pv[7]*y + pv[11]*z + pv[15];

    // skip culled splats
    if (c_w <= 0.f)
    {
        grad_in_pos_x[i] = 0.f; grad_in_pos_y[i] = 0.f; grad_in_pos_z[i] = 0.f;
        grad_in_cov_xx[i] = 0.f; grad_in_cov_xy[i] = 0.f; grad_in_cov_xz[i] = 0.f;
        grad_in_cov_yy[i] = 0.f; grad_in_cov_yz[i] = 0.f; grad_in_cov_zz[i] = 0.f;
        grad_in_color_r[i] = 0.f; grad_in_color_g[i] = 0.f; grad_in_color_b[i] = 0.f;
        grad_in_opacity[i] = 0.f;
        return;
    }

    float inv_w  = 1.f / c_w;
    float inv_w2 = inv_w * inv_w;

    // recompute Jacobian rows
    float jr0_x = (pv[0] * c_w - pv[ 3] * c_x) * inv_w2;
    float jr0_y = (pv[4] * c_w - pv[ 7] * c_x) * inv_w2;
    float jr0_z = (pv[8] * c_w - pv[11] * c_x) * inv_w2;

    float jr1_x = (pv[1] * c_w - pv[ 3] * c_y) * inv_w2;
    float jr1_y = (pv[5] * c_w - pv[ 7] * c_y) * inv_w2;
    float jr1_z = (pv[9] * c_w - pv[11] * c_y) * inv_w2;


    // ===== covariance backward =====
    float d_cxx = grad_cov_xx[i];
    float d_cxy = grad_cov_xy[i];
    float d_cyy = grad_cov_yy[i];

    float sxx = cov_xx[i], sxy = cov_xy[i], sxz = cov_xz[i];
    float                  syy = cov_yy[i], syz = cov_yz[i];
    float                                   szz = cov_zz[i];

    float s_jr0_x = sxx*jr0_x + sxy*jr0_y + sxz*jr0_z;
    float s_jr0_y = sxy*jr0_x + syy*jr0_y + syz*jr0_z;
    float s_jr0_z = sxz*jr0_x + syz*jr0_y + szz*jr0_z;

    float s_jr1_x = sxx*jr1_x + sxy*jr1_y + sxz*jr1_z;
    float s_jr1_y = sxy*jr1_x + syy*jr1_y + syz*jr1_z;
    float s_jr1_z = sxz*jr1_x + syz*jr1_y + szz*jr1_z;

    // dL/dCov_3D = 2*d_cxx*(JR0 outer s_jr0)
    //            + d_cxy*(JR0 outer s_jr1 + JR1 outer s_jr0)
    //            + 2*d_cyy*(JR1 outer s_jr1)
    grad_in_cov_xx[i] = 2.f*d_cxx*jr0_x*s_jr0_x + d_cxy*(jr0_x*s_jr1_x + jr1_x*s_jr0_x) + 2.f*d_cyy*jr1_x*s_jr1_x;
    grad_in_cov_xy[i] = 2.f*d_cxx*jr0_x*s_jr0_y + d_cxy*(jr0_x*s_jr1_y + jr1_x*s_jr0_y) + 2.f*d_cyy*jr1_x*s_jr1_y;
    grad_in_cov_xz[i] = 2.f*d_cxx*jr0_x*s_jr0_z + d_cxy*(jr0_x*s_jr1_z + jr1_x*s_jr0_z) + 2.f*d_cyy*jr1_x*s_jr1_z;
    grad_in_cov_yy[i] = 2.f*d_cxx*jr0_y*s_jr0_y + d_cxy*(jr0_y*s_jr1_y + jr1_y*s_jr0_y) + 2.f*d_cyy*jr1_y*s_jr1_y;
    grad_in_cov_yz[i] = 2.f*d_cxx*jr0_y*s_jr0_z + d_cxy*(jr0_y*s_jr1_z + jr1_y*s_jr0_z) + 2.f*d_cyy*jr1_y*s_jr1_z;
    grad_in_cov_zz[i] = 2.f*d_cxx*jr0_z*s_jr0_z + d_cxy*(jr0_z*s_jr1_z + jr1_z*s_jr0_z) + 2.f*d_cyy*jr1_z*s_jr1_z;

    // ===== position backward =====
    // ndc = clip / c_w, so d(ndc)/d(world) = J (already computed above)
    // dL/d(world) = J^T * dL/d(ndc)
    float d_ndc_x = grad_pos_x[i];
    float d_ndc_y = grad_pos_y[i];

    grad_in_pos_x[i] = d_ndc_x * jr0_x + d_ndc_y * jr1_x;
    grad_in_pos_y[i] = d_ndc_x * jr0_y + d_ndc_y * jr1_y;
    grad_in_pos_z[i] = d_ndc_x * jr0_z + d_ndc_y * jr1_z;

    // pass-through gradients
    grad_in_color_r[i] = grad_color_r[i];
    grad_in_color_g[i] = grad_color_g[i];
    grad_in_color_b[i] = grad_color_b[i];
    grad_in_opacity[i] = grad_opacity[i];
}

/* ===== ===== Lifecycle ===== ===== */

void PerspProjectLayer::allocate(int count)
{
    allocatedCount = count;
    output.allocateDeviceMem(count);
    gradInput.allocateDeviceMem(count);
    cudaMalloc(&d_pv, 16 * sizeof(float));
}

void PerspProjectLayer::free()
{
    output.free();
    gradInput.free();
    CUDA_FREE(d_pv);
    allocatedCount = 0;
}

void PerspProjectLayer::zero_grad()
{
    gradInput.zero_grad();
}

void PerspProjectLayer::setCamera(const glm::mat4 &view, const glm::mat4 &proj)
{
    glm::mat4 pv = proj * view;
    memcpy(h_pv, glm::value_ptr(pv), 16 * sizeof(float));
    cudaMemcpy(d_pv, h_pv, 16 * sizeof(float), cudaMemcpyHostToDevice);
}

/* ===== ===== Forward / Backward ===== ===== */

void PerspProjectLayer::forward()
{
    int count   = input->count;
    int threads = 256;
    int blocks  = (count + threads - 1) / threads;

    perspProjectForwardKernel<<<blocks, threads>>>(
        input->pos_x,   input->pos_y,   input->pos_z,
        input->cov_xx,  input->cov_xy,  input->cov_xz,
        input->cov_yy,  input->cov_yz,  input->cov_zz,
        input->color_r, input->color_g, input->color_b,
        input->opacity,
        d_pv,
        output.pos_x,   output.pos_y,   output.pos_z,
        output.cov_xx,  output.cov_xy,  output.cov_yy,
        output.color_r, output.color_g, output.color_b,
        output.opacity,
        count
    );
    cudaDeviceSynchronize();
}

void PerspProjectLayer::backward()
{
    int count   = input->count;
    int threads = 256;
    int blocks  = (count + threads - 1) / threads;

    perspProjectBackwardKernel<<<blocks, threads>>>(
        input->pos_x,   input->pos_y,   input->pos_z,
        input->cov_xx,  input->cov_xy,  input->cov_xz,
        input->cov_yy,  input->cov_yz,  input->cov_zz,
        d_pv,
        gradOutput->grad_pos_x,   gradOutput->grad_pos_y,
        gradOutput->grad_cov_xx,  gradOutput->grad_cov_xy,  gradOutput->grad_cov_yy,
        gradOutput->grad_color_r, gradOutput->grad_color_g, gradOutput->grad_color_b,
        gradOutput->grad_opacity,
        gradInput.grad_pos_x,   gradInput.grad_pos_y,   gradInput.grad_pos_z,
        gradInput.grad_cov_xx,  gradInput.grad_cov_xy,  gradInput.grad_cov_xz,
        gradInput.grad_cov_yy,  gradInput.grad_cov_yz,  gradInput.grad_cov_zz,
        gradInput.grad_color_r, gradInput.grad_color_g, gradInput.grad_color_b,
        gradInput.grad_opacity,
        count
    );
    cudaDeviceSynchronize();
}
