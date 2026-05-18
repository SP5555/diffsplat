#include "veg_activ_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
#include "../cuda/math_ops.cuh"

/* ===== ===== Kernel ===== ===== */

// Splats below this opacity contribute nothing at 8-bit precision.
static constexpr float VEG_OPACITY_CULL = 1.f / 255.f;

/**
 * @brief VEG activation: geometry (same as GaussActivLayer) + LUT-based color.
 *
 * Each thread handles one Gaussian:
 *   - Evaluates opacity from LUT + sigmoid; skips the splat if below cull threshold.
 *   - Surviving splats claim a compacted output slot via atomicAdd.
 *   - Builds 3D covariance Cov = R*S*S^T*R^T from log-scale + quaternion.
 *   - Maps logit_scalar through sigmoid -> 256-entry RGB LUT (bilinear).
 */
__global__ void vegActivKernel(
    // inputs
    const float *__restrict__ i_px,
    const float *__restrict__ i_py,
    const float *__restrict__ i_pz,
    const float *__restrict__ i_sx,
    const float *__restrict__ i_sy,
    const float *__restrict__ i_sz,
    const float *__restrict__ i_rw,
    const float *__restrict__ i_rx,
    const float *__restrict__ i_ry,
    const float *__restrict__ i_rz,
    const float *__restrict__ i_logit_opacity,
    const float *__restrict__ i_logit_scalar,
    // LUT (device)
    const float *__restrict__ lut_rgb,   // VEG_LUT_SIZE * 3
    const float *__restrict__ lut_alpha, // VEG_LUT_SIZE
    // outputs
    float    *o_px,   float *o_py,   float *o_pz,
    float    *o_cxx,  float *o_cxy,  float *o_cxz,
    float    *o_cyy,  float *o_cyz,  float *o_czz,
    float    *o_r,    float *o_g,    float *o_b,
    float    *o_a,
    uint32_t *o_count,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // LUT lookup first -- cheap, and gates all the geometry work below.
    float scalar_01 = sigmoid(i_logit_scalar[i]);
    float idx_f     = scalar_01 * (float)(VEG_LUT_SIZE - 1);
    int   lo        = (int)idx_f;
    int   hi        = min(lo + 1, VEG_LUT_SIZE - 1);
    float t         = idx_f - (float)lo;

    float tf_alpha = lut_alpha[lo] * (1.f-t) + lut_alpha[hi] * t;
    float opacity  = tf_alpha * sigmoid(i_logit_opacity[i]);

    if (opacity < VEG_OPACITY_CULL) return;

    int slot = (int)atomicAdd(o_count, 1u);

    // position pass-through
    o_px[slot] = i_px[i];
    o_py[slot] = i_py[i];
    o_pz[slot] = i_pz[i];

    // normalize quaternion -> rotation matrix
    NormQuat nq = normalizeQuat(i_rw[i], i_rx[i], i_ry[i], i_rz[i]);
    RotMat3  R  = quatToRotMat(nq.w, nq.x, nq.y, nq.z);

    // actual scale
    float sx = expf(i_sx[i]);
    float sy = expf(i_sy[i]);
    float sz = expf(i_sz[i]);

    // M = R * S
    float m00 = R.r00*sx, m01 = R.r01*sy, m02 = R.r02*sz;
    float m10 = R.r10*sx, m11 = R.r11*sy, m12 = R.r12*sz;
    float m20 = R.r20*sx, m21 = R.r21*sy, m22 = R.r22*sz;

    // Cov = M * M^T (upper triangle)
    o_cxx[slot] = m00*m00 + m01*m01 + m02*m02;
    o_cxy[slot] = m00*m10 + m01*m11 + m02*m12;
    o_cxz[slot] = m00*m20 + m01*m21 + m02*m22;
    o_cyy[slot] = m10*m10 + m11*m11 + m12*m12;
    o_cyz[slot] = m10*m20 + m11*m21 + m12*m22;
    o_czz[slot] = m20*m20 + m21*m21 + m22*m22;

    // color from LUT
    o_r[slot] = lut_rgb[lo*3+0] * (1.f-t) + lut_rgb[hi*3+0] * t;
    o_g[slot] = lut_rgb[lo*3+1] * (1.f-t) + lut_rgb[hi*3+1] * t;
    o_b[slot] = lut_rgb[lo*3+2] * (1.f-t) + lut_rgb[hi*3+2] * t;
    o_a[slot] = opacity;
}

/* ===== ===== Layer methods ===== ===== */

void VegActivLayer::allocate(int n)
{
    output.allocate(n);

    d_lut_rgb.allocate(VEG_LUT_SIZE * 3);
    d_lut_alpha.allocate(VEG_LUT_SIZE);
    d_out_count.allocate(1);
    initDefaultLUT();
}

void VegActivLayer::initDefaultLUT()
{
    float rgb[VEG_LUT_SIZE * 3];
    float alpha[VEG_LUT_SIZE];
    for (int i = 0; i < VEG_LUT_SIZE; i++)
    {
        float v = (float)i / (float)(VEG_LUT_SIZE - 1);
        rgb[i*3+0] = v;
        rgb[i*3+1] = v;
        rgb[i*3+2] = v;
        alpha[i]   = v;
    }
    setLUT(rgb, alpha);
}

void VegActivLayer::setLUT(const float *rgb, const float *alpha)
{
    if (!d_lut_rgb.ptr || !d_lut_alpha.ptr) return;
    CUDA_CHECK(cudaMemcpy(d_lut_rgb,   rgb,   VEG_LUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lut_alpha, alpha, VEG_LUT_SIZE     * sizeof(float), cudaMemcpyHostToDevice));
}

void VegActivLayer::forward()
{
    int count = input->count;

    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(uint32_t)));

    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vegActivKernel<<<blocks, BLOCK_SIZE>>>(
        input->pos_x,   input->pos_y,   input->pos_z,
        input->scale_x, input->scale_y, input->scale_z,
        input->rot_w,   input->rot_x,   input->rot_y,   input->rot_z,
        input->logit_opacity,
        input->logit_scalar,
        d_lut_rgb, d_lut_alpha,
        output.pos_x,   output.pos_y,   output.pos_z,
        output.cov_xx,  output.cov_xy,  output.cov_xz,
        output.cov_yy,  output.cov_yz,  output.cov_zz,
        output.color_r, output.color_g, output.color_b,
        output.opacity,
        d_out_count,
        count
    );
    CUDA_SYNC_CHECK();

    uint32_t surviving = 0;
    CUDA_CHECK(cudaMemcpy(&surviving, d_out_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    output.count = (int)surviving;
}
