#pragma once
#include <cuda_runtime.h>

#include "../cuda/cuda_buffer.h"

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
    CudaBuffer<float> pos_x, pos_y, pos_z;
    CudaBuffer<float> cov_xx, cov_xy, cov_yy;
    CudaBuffer<float> color_r, color_g, color_b;
    CudaBuffer<float> opacity;

    void allocate(int n)
    {
        pos_x.allocate(n);   pos_y.allocate(n);   pos_z.allocate(n);
        cov_xx.allocate(n);  cov_xy.allocate(n);  cov_yy.allocate(n);
        color_r.allocate(n); color_g.allocate(n); color_b.allocate(n);
        opacity.allocate(n);
        count = n;
    }
};

/**
 * @brief Gradients w.r.t. Splat2DParams.
 */
struct Splat2DGrads
{
    int count = 0;

    // NDC space
    CudaBuffer<float> grad_pos_x, grad_pos_y, grad_pos_z;
    CudaBuffer<float> grad_cov_xx, grad_cov_xy, grad_cov_yy;
    CudaBuffer<float> grad_color_r, grad_color_g, grad_color_b;
    CudaBuffer<float> grad_opacity;

    void allocate(int n)
    {
        grad_pos_x.allocate(n);   grad_pos_y.allocate(n);   grad_pos_z.allocate(n);
        grad_cov_xx.allocate(n);  grad_cov_xy.allocate(n);  grad_cov_yy.allocate(n);
        grad_color_r.allocate(n); grad_color_g.allocate(n); grad_color_b.allocate(n);
        grad_opacity.allocate(n);
        count = n;
    }

    void zero_grad()
    {
        grad_pos_x.zero();   grad_pos_y.zero();   grad_pos_z.zero();
        grad_cov_xx.zero();  grad_cov_xy.zero();  grad_cov_yy.zero();
        grad_color_r.zero(); grad_color_g.zero(); grad_color_b.zero();
        grad_opacity.zero();
    }
};
