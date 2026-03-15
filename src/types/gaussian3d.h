#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "../utils/cuda_utils.h"

/* ===== ===== Gaussian3D (CPU) ===== ===== */

/**
 * @brief CPU-side structure for a single 3D Gaussian splat.
 */
struct Gaussian3D
{
    float x, y, z;
    float scale_x, scale_y, scale_z;  // log-scale
    float rot_w, rot_x, rot_y, rot_z; // unit quaternion
    float r, g, b;
    float opacity;
};

/* ===== ===== Gaussian3DParams ===== ===== */

/**
 * @brief GPU-side Gaussian parameters, stored as struct-of-arrays
 * for coalesced memory access.
 */
struct Gaussian3DParams
{
    int count = 0;

    CudaBuffer<float> pos_x,   pos_y,   pos_z;
    CudaBuffer<float> scale_x, scale_y, scale_z;
    CudaBuffer<float> rot_w,   rot_x,   rot_y,   rot_z;
    CudaBuffer<float> color_sh_r, color_sh_g, color_sh_b;
    CudaBuffer<float> logit_opacity;

    void allocate(int n)
    {
        pos_x.allocate(n);   pos_y.allocate(n);   pos_z.allocate(n);
        scale_x.allocate(n); scale_y.allocate(n); scale_z.allocate(n);
        rot_w.allocate(n);   rot_x.allocate(n);   rot_y.allocate(n);   rot_z.allocate(n);
        color_sh_r.allocate(n);
        color_sh_g.allocate(n);
        color_sh_b.allocate(n);
        logit_opacity.allocate(n);
        count = n;
    }

    void upload(const std::vector<Gaussian3D> &host)
    {
        int n = (int)host.size();
        allocate(n);

        std::vector<float> tmp(n);
        auto up = [&](float *dst, auto getter) {
            for (int i = 0; i < n; i++) tmp[i] = getter(host[i]);
            cudaMemcpy(dst, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        };

        up(pos_x,         [](const Gaussian3D &g) { return g.x; });
        up(pos_y,         [](const Gaussian3D &g) { return g.y; });
        up(pos_z,         [](const Gaussian3D &g) { return g.z; });
        up(scale_x,       [](const Gaussian3D &g) { return g.scale_x; });
        up(scale_y,       [](const Gaussian3D &g) { return g.scale_y; });
        up(scale_z,       [](const Gaussian3D &g) { return g.scale_z; });
        up(rot_w,         [](const Gaussian3D &g) { return g.rot_w; });
        up(rot_x,         [](const Gaussian3D &g) { return g.rot_x; });
        up(rot_y,         [](const Gaussian3D &g) { return g.rot_y; });
        up(rot_z,         [](const Gaussian3D &g) { return g.rot_z; });
        up(color_sh_r,    [](const Gaussian3D &g) { return g.r; });
        up(color_sh_g,    [](const Gaussian3D &g) { return g.g; });
        up(color_sh_b,    [](const Gaussian3D &g) { return g.b; });
        up(logit_opacity, [](const Gaussian3D &g) { return g.opacity; });
    }
};

struct Gaussian3DGrads
{
    int count = 0;

    // gradients (zeroed each iteration)
    CudaBuffer<float> grad_pos_x,   grad_pos_y,   grad_pos_z;
    CudaBuffer<float> grad_scale_x, grad_scale_y, grad_scale_z;
    CudaBuffer<float> grad_rot_w,   grad_rot_x,   grad_rot_y,   grad_rot_z;
    CudaBuffer<float> grad_color_sh_r, grad_color_sh_g, grad_color_sh_b;
    CudaBuffer<float> grad_logit_opacity;

    void allocate(int n)
    {
        grad_pos_x.allocate(n);   grad_pos_y.allocate(n);   grad_pos_z.allocate(n);
        grad_scale_x.allocate(n); grad_scale_y.allocate(n); grad_scale_z.allocate(n);
        grad_rot_w.allocate(n);   grad_rot_x.allocate(n);
        grad_rot_y.allocate(n);   grad_rot_z.allocate(n);
        grad_color_sh_r.allocate(n);
        grad_color_sh_g.allocate(n);
        grad_color_sh_b.allocate(n);
        grad_logit_opacity.allocate(n);
        count = n;
    }

    void zero_grad()
    {
        grad_pos_x.zero();   grad_pos_y.zero();   grad_pos_z.zero();
        grad_scale_x.zero(); grad_scale_y.zero(); grad_scale_z.zero();
        grad_rot_w.zero();   grad_rot_x.zero();   grad_rot_y.zero();   grad_rot_z.zero();
        grad_color_sh_r.zero();
        grad_color_sh_g.zero();
        grad_color_sh_b.zero();
        grad_logit_opacity.zero();
    }
};