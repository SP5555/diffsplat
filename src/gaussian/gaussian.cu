#include "gaussian.h"
#include <cstring>
#include <cmath>

static float *deviceAlloc(int n)
{
    float *ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(float));
    cudaMemset(ptr, 0, n * sizeof(float));
    return ptr;
}

void GaussianParams::allocateDeviceMem(int n)
{
    count = n;

    pos_x = deviceAlloc(n); pos_y = deviceAlloc(n); pos_z = deviceAlloc(n);
    cov_a = deviceAlloc(n); cov_b = deviceAlloc(n); cov_d = deviceAlloc(n);
    color_r = deviceAlloc(n); color_g = deviceAlloc(n); color_b = deviceAlloc(n);
    opacity = deviceAlloc(n);
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
    auto f = [](float *p)
    { if (p) cudaFree(p); };
    f(pos_x); f(pos_y); f(pos_z);
    f(cov_a); f(cov_b); f(cov_d);
    f(color_r); f(color_g); f(color_b);
    f(opacity);

    count = 0;
}

GaussianParams GaussianParams::randomInit(int n, int seed)
{
    srand(seed);
    // [-1, 1]
    auto rnd = []()
    { return ((float)rand() / RAND_MAX) * 2.f - 1.f; };
    // [ 0, 1]
    auto rndu = []()
    { return (float)rand() / RAND_MAX; };

    std::vector<Gaussian3D> host(n);
    for (auto &g : host)
    {
        g.x = rnd();
        g.y = rnd();
        g.z = 0.f;       // Z unused in 2D
        g.cov_a = 0.001f; // small isotropic covariance to start
        g.cov_b = 0.f;
        g.cov_d = 0.001f;
        g.r = rndu();
        g.g = rndu();
        g.b = rndu();
        g.opacity = 0.5f + 0.5f * rndu();
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
    auto f = [](float *p)
    { if (p) cudaFree(p); };
    f(grad_pos_x); f(grad_pos_y);
    f(grad_cov_a); f(grad_cov_b); f(grad_cov_d);
    f(grad_color_r); f(grad_color_g); f(grad_color_b);
    f(grad_opacity);

    f(m_pos_x); f(v_pos_x);
    f(m_pos_y); f(v_pos_y);
    f(m_cov_a); f(v_cov_a);
    f(m_cov_b); f(v_cov_b);
    f(m_cov_d); f(v_cov_d);
    f(m_color_r); f(v_color_r);
    f(m_color_g); f(v_color_g);
    f(m_color_b); f(v_color_b);
    f(m_opacity); f(v_opacity);

    count = 0;
}