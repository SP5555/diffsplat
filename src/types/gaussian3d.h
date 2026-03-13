#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "../utils/cuda_utils.cuh"

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
    CudaBuffer<float> color_r, color_g, color_b;
    CudaBuffer<float> opacity;

    void allocate(int n)
    {
        pos_x.allocate(n);   pos_y.allocate(n);   pos_z.allocate(n);
        scale_x.allocate(n); scale_y.allocate(n); scale_z.allocate(n);
        rot_w.allocate(n);   rot_x.allocate(n);   rot_y.allocate(n);   rot_z.allocate(n);
        color_r.allocate(n); color_g.allocate(n); color_b.allocate(n);
        opacity.allocate(n);
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

        up(pos_x,   [](const Gaussian3D &g) { return g.x; });
        up(pos_y,   [](const Gaussian3D &g) { return g.y; });
        up(pos_z,   [](const Gaussian3D &g) { return g.z; });
        up(scale_x, [](const Gaussian3D &g) { return g.scale_x; });
        up(scale_y, [](const Gaussian3D &g) { return g.scale_y; });
        up(scale_z, [](const Gaussian3D &g) { return g.scale_z; });
        up(rot_w,   [](const Gaussian3D &g) { return g.rot_w; });
        up(rot_x,   [](const Gaussian3D &g) { return g.rot_x; });
        up(rot_y,   [](const Gaussian3D &g) { return g.rot_y; });
        up(rot_z,   [](const Gaussian3D &g) { return g.rot_z; });
        up(color_r, [](const Gaussian3D &g) { return g.r; });
        up(color_g, [](const Gaussian3D &g) { return g.g; });
        up(color_b, [](const Gaussian3D &g) { return g.b; });
        up(opacity, [](const Gaussian3D &g) { return g.opacity; });
    }

    static Gaussian3DParams randomInit(int n, int width, int height, int seed = 42)
    {
        srand(seed);
        auto rnd  = []() { return ((float)rand() / RAND_MAX) * 2.f - 1.f; };
        auto rndu = []() { return  (float)rand() / RAND_MAX; };

        float half_w   = (float)width  * 0.5f;
        float half_h   = (float)height * 0.5f;
        float log_sigma = logf(3.f);

        std::vector<Gaussian3D> host(n);
        for (auto &g : host)
        {
            g.x = rnd() * half_w;
            g.y = rnd() * half_h;
            g.z = rnd();

            g.scale_x = log_sigma + rnd() * 0.5f;
            g.scale_y = log_sigma + rnd() * 0.5f;
            g.scale_z = log_sigma + rnd() * 0.5f;

            g.rot_w = 1.f + rnd() * 0.1f;
            g.rot_x =       rnd() * 0.1f;
            g.rot_y =       rnd() * 0.1f;
            g.rot_z =       rnd() * 0.1f;
            float norm = sqrtf(g.rot_w*g.rot_w + g.rot_x*g.rot_x +
                               g.rot_y*g.rot_y + g.rot_z*g.rot_z);
            g.rot_w /= norm; g.rot_x /= norm;
            g.rot_y /= norm; g.rot_z /= norm;

            g.r       = rndu();
            g.g       = rndu();
            g.b       = rndu();
            float o_raw = 0.6f + 0.4f * rndu();
            // logit
            g.opacity = logf(o_raw / (1.f - o_raw));
        }

        Gaussian3DParams data;
        data.upload(host);
        return data;
    }
};

/* ===== ===== Gaussian3DOptState ===== ===== */

/**
 * @brief GPU-side optimization state: gradient buffers and Adam moments.
 */
struct Gaussian3DOptState
{
    int count = 0;

    // gradients (zeroed each iteration)
    CudaBuffer<float> grad_pos_x,   grad_pos_y,   grad_pos_z;
    CudaBuffer<float> grad_scale_x, grad_scale_y, grad_scale_z;
    CudaBuffer<float> grad_rot_w,   grad_rot_x,   grad_rot_y,   grad_rot_z;
    CudaBuffer<float> grad_color_r, grad_color_g, grad_color_b;
    CudaBuffer<float> grad_opacity;

    // Adam first moments
    CudaBuffer<float> m_pos_x,   m_pos_y,   m_pos_z;
    CudaBuffer<float> m_scale_x, m_scale_y, m_scale_z;
    CudaBuffer<float> m_rot_w,   m_rot_x,   m_rot_y,   m_rot_z;
    CudaBuffer<float> m_color_r, m_color_g, m_color_b;
    CudaBuffer<float> m_opacity;

    // Adam second moments
    CudaBuffer<float> v_pos_x,   v_pos_y,   v_pos_z;
    CudaBuffer<float> v_scale_x, v_scale_y, v_scale_z;
    CudaBuffer<float> v_rot_w,   v_rot_x,   v_rot_y,   v_rot_z;
    CudaBuffer<float> v_color_r, v_color_g, v_color_b;
    CudaBuffer<float> v_opacity;

    void allocate(int n)
    {
        grad_pos_x.allocate(n);   grad_pos_y.allocate(n);   grad_pos_z.allocate(n);
        grad_scale_x.allocate(n); grad_scale_y.allocate(n); grad_scale_z.allocate(n);
        grad_rot_w.allocate(n);   grad_rot_x.allocate(n);
        grad_rot_y.allocate(n);   grad_rot_z.allocate(n);
        grad_color_r.allocate(n); grad_color_g.allocate(n); grad_color_b.allocate(n);
        grad_opacity.allocate(n);

        m_pos_x.allocate(n);   v_pos_x.allocate(n);
        m_pos_y.allocate(n);   v_pos_y.allocate(n);
        m_pos_z.allocate(n);   v_pos_z.allocate(n);
        m_scale_x.allocate(n); v_scale_x.allocate(n);
        m_scale_y.allocate(n); v_scale_y.allocate(n);
        m_scale_z.allocate(n); v_scale_z.allocate(n);
        m_rot_w.allocate(n);   v_rot_w.allocate(n);
        m_rot_x.allocate(n);   v_rot_x.allocate(n);
        m_rot_y.allocate(n);   v_rot_y.allocate(n);
        m_rot_z.allocate(n);   v_rot_z.allocate(n);
        m_color_r.allocate(n); v_color_r.allocate(n);
        m_color_g.allocate(n); v_color_g.allocate(n);
        m_color_b.allocate(n); v_color_b.allocate(n);
        m_opacity.allocate(n); v_opacity.allocate(n);

        count = n;
    }

    void zero_grad()
    {
        grad_pos_x.zero();   grad_pos_y.zero();   grad_pos_z.zero();
        grad_scale_x.zero(); grad_scale_y.zero(); grad_scale_z.zero();
        grad_rot_w.zero();   grad_rot_x.zero();   grad_rot_y.zero();   grad_rot_z.zero();
        grad_color_r.zero(); grad_color_g.zero(); grad_color_b.zero();
        grad_opacity.zero();
    }
};
