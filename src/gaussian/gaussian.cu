#include "gaussian.h"
#include <cstring>
#include <cmath>

#include "../utils/cuda_utils.h"

static float *deviceAlloc(int n)
{
    float *ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(float));
    cudaMemset(ptr, 0, n * sizeof(float));
    return ptr;
}

void GaussianParams::allocateDeviceMem(int n)
{
    pos_x = deviceAlloc(n); pos_y = deviceAlloc(n); pos_z = deviceAlloc(n);
    cov_a = deviceAlloc(n); cov_b = deviceAlloc(n); cov_d = deviceAlloc(n);
    color_r = deviceAlloc(n); color_g = deviceAlloc(n); color_b = deviceAlloc(n);
    opacity = deviceAlloc(n);

    count = n;
}

void GaussianParams::upload(const std::vector<Gaussian3D> &host)
{
    int n = (int)host.size();
    allocateDeviceMem(n);

    // Unpack AoS host -> SoA device
    std::vector<float> tmp(n);

    // upload one attribute at a time
    auto up = [&](float *dst, auto getter)
    {
        for (int i = 0; i < n; i++)
            tmp[i] = getter(host[i]);
        cudaMemcpy(dst, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    };

    up(pos_x, [](const Gaussian3D &g) { return g.x; });
    up(pos_y, [](const Gaussian3D &g) { return g.y; });
    up(pos_z, [](const Gaussian3D &g) { return g.z; });
    up(cov_a, [](const Gaussian3D &g) { return g.cov_a; });
    up(cov_b, [](const Gaussian3D &g) { return g.cov_b; });
    up(cov_d, [](const Gaussian3D &g) { return g.cov_d; });
    up(color_r, [](const Gaussian3D &g) { return g.r; });
    up(color_g, [](const Gaussian3D &g) { return g.g; });
    up(color_b, [](const Gaussian3D &g) { return g.b; });
    up(opacity, [](const Gaussian3D &g) { return g.opacity; });
}

void GaussianParams::free()
{
    CUDA_FREE(pos_x); CUDA_FREE(pos_y); CUDA_FREE(pos_z);
    CUDA_FREE(cov_a); CUDA_FREE(cov_b); CUDA_FREE(cov_d);
    CUDA_FREE(color_r); CUDA_FREE(color_g); CUDA_FREE(color_b);
    CUDA_FREE(opacity);

    count = 0;
}

GaussianParams GaussianParams::randomInit(int n, int width, int height, int seed)
{
    srand(seed);
    // [-1, 1]
    auto rnd = []()
    { return ((float)rand() / RAND_MAX) * 2.f - 1.f; };
    // [ 0, 1]
    auto rndu = []()
    { return (float)rand() / RAND_MAX; };

    float invAspect = (float)height / (float)width;
    std::vector<Gaussian3D> host(n);
    for (auto &g : host)
    {
        g.x = rnd();
        g.y = rnd();
        g.z = rnd();
        // small isotropic covariance to start
        g.cov_a = (1e-6f + 6e-5f * rndu()) * invAspect * invAspect;
        g.cov_b = 0.f;
        g.cov_d = 1e-6f + 6e-5f * rndu();
        g.r = rndu();
        g.g = rndu();
        g.b = rndu();
        g.opacity = 0.6f + 0.4f * rndu();
    }

    GaussianParams data;
    data.upload(host);
    return data;
}

void GaussianOptState::allocateDeviceMem(int n)
{
    // for later, I guess
    grad_pos_x = deviceAlloc(n); grad_pos_y = deviceAlloc(n);
    grad_cov_a = deviceAlloc(n); grad_cov_b = deviceAlloc(n); grad_cov_d = deviceAlloc(n);
    grad_color_r = deviceAlloc(n); grad_color_g = deviceAlloc(n); grad_color_b = deviceAlloc(n);
    grad_opacity = deviceAlloc(n);

    m_pos_x = deviceAlloc(n); v_pos_x = deviceAlloc(n);
    m_pos_y = deviceAlloc(n); v_pos_y = deviceAlloc(n);
    m_cov_a = deviceAlloc(n); v_cov_a = deviceAlloc(n);
    m_cov_b = deviceAlloc(n); v_cov_b = deviceAlloc(n);
    m_cov_d = deviceAlloc(n); v_cov_d = deviceAlloc(n);
    m_color_r = deviceAlloc(n); v_color_r = deviceAlloc(n);
    m_color_g = deviceAlloc(n); v_color_g = deviceAlloc(n);
    m_color_b = deviceAlloc(n); v_color_b = deviceAlloc(n);
    m_opacity = deviceAlloc(n); v_opacity = deviceAlloc(n);

    count = n;
}

void GaussianOptState::zeroGradients()
{
    auto z = [&](float *p)
    { if (p) cudaMemset(p, 0, count * sizeof(float)); };
    z(grad_pos_x); z(grad_pos_y);
    z(grad_cov_a); z(grad_cov_b); z(grad_cov_d);
    z(grad_color_r); z(grad_color_g); z(grad_color_b);
    z(grad_opacity);
}

void GaussianOptState::free()
{
    CUDA_FREE(grad_pos_x); CUDA_FREE(grad_pos_y);
    CUDA_FREE(grad_cov_a); CUDA_FREE(grad_cov_b); CUDA_FREE(grad_cov_d);
    CUDA_FREE(grad_color_r); CUDA_FREE(grad_color_g); CUDA_FREE(grad_color_b);
    CUDA_FREE(grad_opacity);

    CUDA_FREE(m_pos_x); CUDA_FREE(v_pos_x);
    CUDA_FREE(m_pos_y); CUDA_FREE(v_pos_y);
    CUDA_FREE(m_cov_a); CUDA_FREE(v_cov_a);
    CUDA_FREE(m_cov_b); CUDA_FREE(v_cov_b);
    CUDA_FREE(m_cov_d); CUDA_FREE(v_cov_d);
    CUDA_FREE(m_color_r); CUDA_FREE(v_color_r);
    CUDA_FREE(m_color_g); CUDA_FREE(v_color_g);
    CUDA_FREE(m_color_b); CUDA_FREE(v_color_b);
    CUDA_FREE(m_opacity); CUDA_FREE(v_opacity);

    count = 0;
}