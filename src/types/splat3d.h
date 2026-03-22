#pragma once
#include <cuda_runtime.h>

#include "../cuda/cuda_buffer.h"

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
    CudaBuffer<float> pos_x,   pos_y,   pos_z;
    CudaBuffer<float> cov_xx,  cov_xy,  cov_xz;
    CudaBuffer<float> cov_yy,  cov_yz,  cov_zz;
    CudaBuffer<float> color_r, color_g, color_b;
    CudaBuffer<float> opacity;

    void allocate(int n)
    {
        pos_x.allocate(n);   pos_y.allocate(n);   pos_z.allocate(n);
        cov_xx.allocate(n);  cov_xy.allocate(n);  cov_xz.allocate(n);
        cov_yy.allocate(n);  cov_yz.allocate(n);  cov_zz.allocate(n);
        color_r.allocate(n); color_g.allocate(n); color_b.allocate(n);
        opacity.allocate(n);
        count = n;
    }
};

/**
 * @brief Gradients w.r.t. Splat3DParams.
 */
struct Splat3DGrads
{
    int count = 0;

    // world space
    CudaBuffer<float> grad_pos_x,   grad_pos_y,   grad_pos_z;
    CudaBuffer<float> grad_cov_xx,  grad_cov_xy,  grad_cov_xz;
    CudaBuffer<float> grad_cov_yy,  grad_cov_yz,  grad_cov_zz;
    CudaBuffer<float> grad_color_r, grad_color_g, grad_color_b;
    CudaBuffer<float> grad_opacity;

    void allocate(int n)
    {
        grad_pos_x.allocate(n);   grad_pos_y.allocate(n);   grad_pos_z.allocate(n);
        grad_cov_xx.allocate(n);  grad_cov_xy.allocate(n);  grad_cov_xz.allocate(n);
        grad_cov_yy.allocate(n);  grad_cov_yz.allocate(n);  grad_cov_zz.allocate(n);
        grad_color_r.allocate(n); grad_color_g.allocate(n); grad_color_b.allocate(n);
        grad_opacity.allocate(n);
        count = n;
    }

    void zero_grad()
    {
        grad_pos_x.zero();   grad_pos_y.zero();   grad_pos_z.zero();
        grad_cov_xx.zero();  grad_cov_xy.zero();  grad_cov_xz.zero();
        grad_cov_yy.zero();  grad_cov_yz.zero();  grad_cov_zz.zero();
        grad_color_r.zero(); grad_color_g.zero(); grad_color_b.zero();
        grad_opacity.zero();
    }
};
