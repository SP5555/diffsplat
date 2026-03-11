#pragma once
#include <cuda_runtime.h>

#include "../utils/cuda_utils.cuh"

namespace splat2d_detail {

inline float *deviceAlloc(int n)
{
    float *ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(float));
    cudaMemset(ptr, 0, n * sizeof(float));
    return ptr;
}

} // namespace splat2d_detail

/**
 * @brief GPU-side SoA storage for 2D Gaussian splats in NDC space.
 *
 * The 2D covariance is symmetric, stored as upper triangle:
 *   [[cov_xx, cov_xy],
 *    [cov_xy, cov_yy]]
 */
struct Splat2DParams
{
    int count = 0;

    // NDC space
    float *pos_x  = nullptr, *pos_y  = nullptr, *pos_z  = nullptr;
    float *cov_xx = nullptr, *cov_xy = nullptr, *cov_yy = nullptr;

    float *color_r = nullptr, *color_g = nullptr, *color_b = nullptr;
    float *opacity = nullptr;

    void allocateDeviceMem(int n)
    {
        using namespace splat2d_detail;
        pos_x   = deviceAlloc(n); pos_y   = deviceAlloc(n); pos_z = deviceAlloc(n);
        cov_xx  = deviceAlloc(n); cov_xy  = deviceAlloc(n); cov_yy = deviceAlloc(n);
        color_r = deviceAlloc(n); color_g = deviceAlloc(n); color_b = deviceAlloc(n);
        opacity = deviceAlloc(n);
        count = n;
    }

    void free()
    {
        CUDA_FREE(pos_x);   CUDA_FREE(pos_y);   CUDA_FREE(pos_z);
        CUDA_FREE(cov_xx);  CUDA_FREE(cov_xy);  CUDA_FREE(cov_yy);
        CUDA_FREE(color_r); CUDA_FREE(color_g); CUDA_FREE(color_b);
        CUDA_FREE(opacity);
        count = 0;
    }
};

/**
 * @brief Gradients w.r.t. Splat2DParams.
 */
struct Splat2DGrads
{
    int count = 0;

    // NDC space
    float *grad_pos_x = nullptr, *grad_pos_y = nullptr;
    float *grad_cov_xx = nullptr, *grad_cov_xy = nullptr, *grad_cov_yy = nullptr;

    float *grad_color_r = nullptr, *grad_color_g = nullptr, *grad_color_b = nullptr;
    float *grad_opacity = nullptr;

    void allocateDeviceMem(int n)
    {
        using namespace splat2d_detail;
        grad_pos_x   = deviceAlloc(n); grad_pos_y   = deviceAlloc(n);
        grad_cov_xx  = deviceAlloc(n); grad_cov_xy  = deviceAlloc(n); grad_cov_yy = deviceAlloc(n);
        grad_color_r = deviceAlloc(n); grad_color_g = deviceAlloc(n); grad_color_b = deviceAlloc(n);
        grad_opacity = deviceAlloc(n);
        count = n;
    }

    void zero()
    {
        auto z = [&](float *p) { if (p) cudaMemset(p, 0, count * sizeof(float)); };
        z(grad_pos_x);   z(grad_pos_y);
        z(grad_cov_xx);  z(grad_cov_xy);  z(grad_cov_yy);
        z(grad_color_r); z(grad_color_g); z(grad_color_b);
        z(grad_opacity);
    }

    void free()
    {
        CUDA_FREE(grad_pos_x);   CUDA_FREE(grad_pos_y);
        CUDA_FREE(grad_cov_xx);  CUDA_FREE(grad_cov_xy);  CUDA_FREE(grad_cov_yy);
        CUDA_FREE(grad_color_r); CUDA_FREE(grad_color_g); CUDA_FREE(grad_color_b);
        CUDA_FREE(grad_opacity);
        count = 0;
    }
};
