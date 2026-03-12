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

/* ===== ===== Helpers ===== ===== */

namespace gaussian3d_detail {

inline float *deviceAlloc(int n)
{
    float *ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(float));
    cudaMemset(ptr, 0, n * sizeof(float));
    return ptr;
}

} // namespace gaussian3d_detail

/* ===== ===== Gaussian3DParams ===== ===== */

/**
 * @brief GPU-side Gaussian parameters, stored as struct-of-arrays
 * for coalesced memory access.
 */
struct Gaussian3DParams
{
    int count = 0;

    float *pos_x   = nullptr, *pos_y   = nullptr, *pos_z   = nullptr;
    float *scale_x = nullptr, *scale_y = nullptr, *scale_z = nullptr;
    float *rot_w   = nullptr, *rot_x   = nullptr, *rot_y   = nullptr, *rot_z   = nullptr;
    float *color_r = nullptr, *color_g = nullptr, *color_b = nullptr;
    float *opacity = nullptr;

    void allocateDeviceMem(int n)
    {
        using namespace gaussian3d_detail;
        pos_x   = deviceAlloc(n); pos_y   = deviceAlloc(n); pos_z   = deviceAlloc(n);
        scale_x = deviceAlloc(n); scale_y = deviceAlloc(n); scale_z = deviceAlloc(n);
        rot_w   = deviceAlloc(n); rot_x   = deviceAlloc(n); rot_y   = deviceAlloc(n); rot_z   = deviceAlloc(n);
        color_r = deviceAlloc(n); color_g = deviceAlloc(n); color_b = deviceAlloc(n);
        opacity = deviceAlloc(n);
        count = n;
    }

    void upload(const std::vector<Gaussian3D> &host)
    {
        int n = (int)host.size();
        allocateDeviceMem(n);

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

    void free()
    {
        CUDA_FREE(pos_x);   CUDA_FREE(pos_y);   CUDA_FREE(pos_z);
        CUDA_FREE(scale_x); CUDA_FREE(scale_y); CUDA_FREE(scale_z);
        CUDA_FREE(rot_w);   CUDA_FREE(rot_x);
        CUDA_FREE(rot_y);   CUDA_FREE(rot_z);
        CUDA_FREE(color_r); CUDA_FREE(color_g); CUDA_FREE(color_b);
        CUDA_FREE(opacity);
        count = 0;
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
    float *grad_pos_x = nullptr, *grad_pos_y = nullptr, *grad_pos_z = nullptr;
    float *grad_scale_x = nullptr, *grad_scale_y = nullptr, *grad_scale_z = nullptr;
    float *grad_rot_w = nullptr, *grad_rot_x = nullptr;
    float *grad_rot_y = nullptr, *grad_rot_z = nullptr;
    float *grad_color_r = nullptr, *grad_color_g = nullptr, *grad_color_b = nullptr;
    float *grad_opacity = nullptr;

    // Adam first moments
    float *m_pos_x = nullptr, *m_pos_y = nullptr, *m_pos_z = nullptr;
    float *m_scale_x = nullptr, *m_scale_y = nullptr, *m_scale_z = nullptr;
    float *m_rot_w = nullptr, *m_rot_x = nullptr, *m_rot_y = nullptr, *m_rot_z = nullptr;
    float *m_color_r = nullptr, *m_color_g = nullptr, *m_color_b = nullptr;
    float *m_opacity = nullptr;

    // Adam second moments
    float *v_pos_x = nullptr, *v_pos_y = nullptr, *v_pos_z = nullptr;
    float *v_scale_x = nullptr, *v_scale_y = nullptr, *v_scale_z = nullptr;
    float *v_rot_w = nullptr, *v_rot_x = nullptr, *v_rot_y = nullptr, *v_rot_z = nullptr;
    float *v_color_r = nullptr, *v_color_g = nullptr, *v_color_b = nullptr;
    float *v_opacity = nullptr;

    void allocateDeviceMem(int n)
    {
        using namespace gaussian3d_detail;
        grad_pos_x   = deviceAlloc(n); grad_pos_y   = deviceAlloc(n); grad_pos_z   = deviceAlloc(n);
        grad_scale_x = deviceAlloc(n); grad_scale_y = deviceAlloc(n); grad_scale_z = deviceAlloc(n);
        grad_rot_w   = deviceAlloc(n); grad_rot_x   = deviceAlloc(n);
        grad_rot_y   = deviceAlloc(n); grad_rot_z   = deviceAlloc(n);
        grad_color_r = deviceAlloc(n); grad_color_g = deviceAlloc(n); grad_color_b = deviceAlloc(n);
        grad_opacity = deviceAlloc(n);

        m_pos_x   = deviceAlloc(n); v_pos_x   = deviceAlloc(n);
        m_pos_y   = deviceAlloc(n); v_pos_y   = deviceAlloc(n);
        m_pos_z   = deviceAlloc(n); v_pos_z   = deviceAlloc(n);
        m_scale_x = deviceAlloc(n); v_scale_x = deviceAlloc(n);
        m_scale_y = deviceAlloc(n); v_scale_y = deviceAlloc(n);
        m_scale_z = deviceAlloc(n); v_scale_z = deviceAlloc(n);
        m_rot_w   = deviceAlloc(n); v_rot_w   = deviceAlloc(n);
        m_rot_x   = deviceAlloc(n); v_rot_x   = deviceAlloc(n);
        m_rot_y   = deviceAlloc(n); v_rot_y   = deviceAlloc(n);
        m_rot_z   = deviceAlloc(n); v_rot_z   = deviceAlloc(n);
        m_color_r = deviceAlloc(n); v_color_r = deviceAlloc(n);
        m_color_g = deviceAlloc(n); v_color_g = deviceAlloc(n);
        m_color_b = deviceAlloc(n); v_color_b = deviceAlloc(n);
        m_opacity = deviceAlloc(n); v_opacity = deviceAlloc(n);

        count = n;
    }

    void zero_grad()
    {
        auto z = [&](float *p) { if (p) cudaMemset(p, 0, count * sizeof(float)); };
        z(grad_pos_x);   z(grad_pos_y);   z(grad_pos_z);
        z(grad_scale_x); z(grad_scale_y); z(grad_scale_z);
        z(grad_rot_w);   z(grad_rot_x);   z(grad_rot_y);   z(grad_rot_z);
        z(grad_color_r); z(grad_color_g); z(grad_color_b);
        z(grad_opacity);
    }

    void free()
    {
        CUDA_FREE(grad_pos_x);   CUDA_FREE(grad_pos_y);   CUDA_FREE(grad_pos_z);
        CUDA_FREE(grad_scale_x); CUDA_FREE(grad_scale_y); CUDA_FREE(grad_scale_z);
        CUDA_FREE(grad_rot_w);   CUDA_FREE(grad_rot_x);
        CUDA_FREE(grad_rot_y);   CUDA_FREE(grad_rot_z);
        CUDA_FREE(grad_color_r); CUDA_FREE(grad_color_g); CUDA_FREE(grad_color_b);
        CUDA_FREE(grad_opacity);

        CUDA_FREE(m_pos_x);   CUDA_FREE(v_pos_x);
        CUDA_FREE(m_pos_y);   CUDA_FREE(v_pos_y);
        CUDA_FREE(m_pos_z);   CUDA_FREE(v_pos_z);
        CUDA_FREE(m_scale_x); CUDA_FREE(v_scale_x);
        CUDA_FREE(m_scale_y); CUDA_FREE(v_scale_y);
        CUDA_FREE(m_scale_z); CUDA_FREE(v_scale_z);
        CUDA_FREE(m_rot_w);   CUDA_FREE(v_rot_w);
        CUDA_FREE(m_rot_x);   CUDA_FREE(v_rot_x);
        CUDA_FREE(m_rot_y);   CUDA_FREE(v_rot_y);
        CUDA_FREE(m_rot_z);   CUDA_FREE(v_rot_z);
        CUDA_FREE(m_color_r); CUDA_FREE(v_color_r);
        CUDA_FREE(m_color_g); CUDA_FREE(v_color_g);
        CUDA_FREE(m_color_b); CUDA_FREE(v_color_b);
        CUDA_FREE(m_opacity); CUDA_FREE(v_opacity);

        count = 0;
    }
};
