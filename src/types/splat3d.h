#pragma once
#include <cuda_runtime.h>

#include "../utils/cuda_utils.cuh"

namespace splat3d_detail {

inline float *deviceAlloc(int n)
{
    float *ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(float));
    cudaMemset(ptr, 0, n * sizeof(float));
    return ptr;
}

} // namespace splat3d_detail

/**
 * @brief GPU-side SoA storage for 3D Gaussian splats in world space.
 *
 * The 3D covariance is symmetric, stored as upper triangle:
 *   [[cov_xx, cov_xy, cov_xz],
 *    [cov_xy, cov_yy, cov_yz],
 *    [cov_xz, cov_yz, cov_zz]]
 */
struct Splat3DParams
{
    int count = 0;

    // world space
    float *pos_x  = nullptr, *pos_y  = nullptr, *pos_z  = nullptr;
    float *cov_xx = nullptr, *cov_xy = nullptr, *cov_xz = nullptr;
    float *cov_yy = nullptr, *cov_yz = nullptr, *cov_zz = nullptr;

    float *color_r = nullptr, *color_g = nullptr, *color_b = nullptr;
    float *opacity = nullptr;

    void allocateDeviceMem(int n)
    {
        using namespace splat3d_detail;
        pos_x   = deviceAlloc(n); pos_y   = deviceAlloc(n); pos_z   = deviceAlloc(n);
        cov_xx  = deviceAlloc(n); cov_xy  = deviceAlloc(n); cov_xz  = deviceAlloc(n);
        cov_yy  = deviceAlloc(n); cov_yz  = deviceAlloc(n); cov_zz  = deviceAlloc(n);
        color_r = deviceAlloc(n); color_g = deviceAlloc(n); color_b = deviceAlloc(n);
        opacity = deviceAlloc(n);
        count = n;
    }

    void free()
    {
        CUDA_FREE(pos_x);   CUDA_FREE(pos_y);   CUDA_FREE(pos_z);
        CUDA_FREE(cov_xx);  CUDA_FREE(cov_xy);  CUDA_FREE(cov_xz);
        CUDA_FREE(cov_yy);  CUDA_FREE(cov_yz);  CUDA_FREE(cov_zz);
        CUDA_FREE(color_r); CUDA_FREE(color_g); CUDA_FREE(color_b);
        CUDA_FREE(opacity);
        count = 0;
    }
};

/**
 * @brief Gradients w.r.t. Splat3DParams.
 */
struct Splat3DGrads
{
    int count = 0;

    // world space
    float *grad_pos_x  = nullptr, *grad_pos_y  = nullptr, *grad_pos_z  = nullptr;
    float *grad_cov_xx = nullptr, *grad_cov_xy = nullptr, *grad_cov_xz = nullptr;
    float *grad_cov_yy = nullptr, *grad_cov_yz = nullptr, *grad_cov_zz = nullptr;

    float *grad_color_r = nullptr, *grad_color_g = nullptr, *grad_color_b = nullptr;
    float *grad_opacity = nullptr;

    void allocateDeviceMem(int n)
    {
        using namespace splat3d_detail;
        grad_pos_x   = deviceAlloc(n); grad_pos_y   = deviceAlloc(n); grad_pos_z   = deviceAlloc(n);
        grad_cov_xx  = deviceAlloc(n); grad_cov_xy  = deviceAlloc(n); grad_cov_xz  = deviceAlloc(n);
        grad_cov_yy  = deviceAlloc(n); grad_cov_yz  = deviceAlloc(n); grad_cov_zz  = deviceAlloc(n);
        grad_color_r = deviceAlloc(n); grad_color_g = deviceAlloc(n); grad_color_b = deviceAlloc(n);
        grad_opacity = deviceAlloc(n);
        count = n;
    }

    void zero_grad()
    {
        auto z = [&](float *p) { if (p) cudaMemset(p, 0, count * sizeof(float)); };
        z(grad_pos_x);   z(grad_pos_y);   z(grad_pos_z);
        z(grad_cov_xx);  z(grad_cov_xy);  z(grad_cov_xz);
        z(grad_cov_yy);  z(grad_cov_yz);  z(grad_cov_zz);
        z(grad_color_r); z(grad_color_g); z(grad_color_b);
        z(grad_opacity);
    }

    void free()
    {
        CUDA_FREE(grad_pos_x);   CUDA_FREE(grad_pos_y);   CUDA_FREE(grad_pos_z);
        CUDA_FREE(grad_cov_xx);  CUDA_FREE(grad_cov_xy);  CUDA_FREE(grad_cov_xz);
        CUDA_FREE(grad_cov_yy);  CUDA_FREE(grad_cov_yz);  CUDA_FREE(grad_cov_zz);
        CUDA_FREE(grad_color_r); CUDA_FREE(grad_color_g); CUDA_FREE(grad_color_b);
        CUDA_FREE(grad_opacity);
        count = 0;
    }
};
