#pragma once
#include <cuda_runtime.h>

#include "../utils/cuda_utils.cuh"

/**
 * NDC-space 2D Gaussian parameters.
 */
struct Splat2DParams
{
    int count = 0;
    
    float *pos_x = nullptr; // [N]
    float *pos_y = nullptr; // [N]
    float *pos_z = nullptr; // [N]

    float *cov_a = nullptr; // [N]
    float *cov_b = nullptr; // [N]
    float *cov_d = nullptr; // [N]
    
    float *color_r = nullptr; // [N]
    float *color_g = nullptr; // [N]
    float *color_b = nullptr; // [N]
    float *opacity = nullptr; // [N]
};

/**
 * Gradients w.r.t. Splat2DParams (NDC space).
 */
struct Splat2DGrads
{
    int count = 0;

    float *pos_x   = nullptr; // [N]
    float *pos_y   = nullptr; // [N]
    float *cov_a   = nullptr; // [N]
    float *cov_b   = nullptr; // [N]
    float *cov_d   = nullptr; // [N]
    float *color_r = nullptr; // [N]
    float *color_g = nullptr; // [N]
    float *color_b = nullptr; // [N]
    float *opacity = nullptr; // [N]

    void allocate(int n)
    {
        auto alloc = [](int n) {
            float *p = nullptr;
            cudaMalloc(&p, n * sizeof(float));
            cudaMemset(p, 0, n * sizeof(float));
            return p;
        };
        pos_x   = alloc(n); pos_y   = alloc(n);
        cov_a   = alloc(n); cov_b   = alloc(n); cov_d = alloc(n);
        color_r = alloc(n); color_g = alloc(n); color_b = alloc(n);
        opacity = alloc(n);
        count = n;
    }

    void free()
    {
        CUDA_FREE(pos_x); CUDA_FREE(pos_y);
        CUDA_FREE(cov_a); CUDA_FREE(cov_b); CUDA_FREE(cov_d);
        CUDA_FREE(color_r); CUDA_FREE(color_g); CUDA_FREE(color_b);
        CUDA_FREE(opacity);
        count = 0;
    }

    void zero()
    {
        auto z = [&](float *p) { if (p) cudaMemset(p, 0, count * sizeof(float)); };
        z(pos_x);   z(pos_y);
        z(cov_a);   z(cov_b);  z(cov_d);
        z(color_r); z(color_g); z(color_b);
        z(opacity);
    }
};
