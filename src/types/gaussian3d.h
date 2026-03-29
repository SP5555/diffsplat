#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../cuda/cuda_buffer.h"
#include "../utils/sh_consts.h"

/* ===== ===== Gaussian3D (CPU) ===== ===== */

/**
 * @brief CPU-side structure for a single 3D Gaussian splat.
 */
struct Gaussian3D
{
    float x, y, z;
    float scale_x, scale_y, scale_z;  // log-scale
    float rot_w, rot_x, rot_y, rot_z; // unit quaternion
    float r, g, b;                     // DC SH coefficients (degree 0)
    float sh_rest[45] = {};            // higher-order SH, bands 0-14 x 3 channels (channel-first: R then G then B)
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

    // Higher-order SH: flat buffers of size n * sh_num_bands each.
    // Layout within each buffer: [band0_splat0..splat(n-1) | band1_splat0.. | ...]
    // Access in kernels: sh_rest_r[band * count + i]
    CudaBuffer<float> sh_rest_r, sh_rest_g, sh_rest_b;
    int sh_num_bands = 0;

    void allocate(int n, int sh_degree = 0)
    {
        pos_x.allocate(n);   pos_y.allocate(n);   pos_z.allocate(n);
        scale_x.allocate(n); scale_y.allocate(n); scale_z.allocate(n);
        rot_w.allocate(n);   rot_x.allocate(n);   rot_y.allocate(n);   rot_z.allocate(n);
        color_sh_r.allocate(n);
        color_sh_g.allocate(n);
        color_sh_b.allocate(n);
        logit_opacity.allocate(n);

        sh_num_bands = sh_degree_to_bands(sh_degree);
        if (sh_num_bands > 0) {
            sh_rest_r.allocate(n * sh_num_bands);
            sh_rest_g.allocate(n * sh_num_bands);
            sh_rest_b.allocate(n * sh_num_bands);
        }
        count = n;
    }

    void upload(const std::vector<Gaussian3D> &host, int sh_degree = 0)
    {
        int n = (int)host.size();
        allocate(n, sh_degree);

        std::vector<float> tmp(n);
        auto up = [&](float *dst, auto getter) {
            for (int i = 0; i < n; i++) tmp[i] = getter(host[i]);
            CUDA_CHECK(cudaMemcpy(dst, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
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

        // PLY channel-first layout: sh_rest[0..K-1] = R bands, [K..2K-1] = G, [2K..3K-1] = B
        // Our GPU layout: sh_rest_r[band * n + i], so copy band-by-band.
        if (sh_num_bands > 0) {
            std::vector<float> band(n);
            for (int b = 0; b < sh_num_bands; b++) {
                // R: PLY index b, G: b + K, B: b + 2*K
                for (int i = 0; i < n; i++) band[i] = host[i].sh_rest[b];
                CUDA_CHECK(cudaMemcpy(sh_rest_r.ptr + b * n, band.data(), n * sizeof(float), cudaMemcpyHostToDevice));

                for (int i = 0; i < n; i++) band[i] = host[i].sh_rest[sh_num_bands + b];
                CUDA_CHECK(cudaMemcpy(sh_rest_g.ptr + b * n, band.data(), n * sizeof(float), cudaMemcpyHostToDevice));

                for (int i = 0; i < n; i++) band[i] = host[i].sh_rest[2 * sh_num_bands + b];
                CUDA_CHECK(cudaMemcpy(sh_rest_b.ptr + b * n, band.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    std::vector<Gaussian3D> download() const
    {
        int n = count;
        std::vector<Gaussian3D> host(n);
        std::vector<float> tmp(n);

        auto dn = [&](const float *src, auto setter) {
            CUDA_CHECK(cudaMemcpy(tmp.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++) setter(host[i], tmp[i]);
        };

        dn(pos_x,         [](Gaussian3D &g, float v) { g.x       = v; });
        dn(pos_y,         [](Gaussian3D &g, float v) { g.y       = v; });
        dn(pos_z,         [](Gaussian3D &g, float v) { g.z       = v; });
        dn(scale_x,       [](Gaussian3D &g, float v) { g.scale_x = v; });
        dn(scale_y,       [](Gaussian3D &g, float v) { g.scale_y = v; });
        dn(scale_z,       [](Gaussian3D &g, float v) { g.scale_z = v; });
        dn(rot_w,         [](Gaussian3D &g, float v) { g.rot_w   = v; });
        dn(rot_x,         [](Gaussian3D &g, float v) { g.rot_x   = v; });
        dn(rot_y,         [](Gaussian3D &g, float v) { g.rot_y   = v; });
        dn(rot_z,         [](Gaussian3D &g, float v) { g.rot_z   = v; });
        dn(color_sh_r,    [](Gaussian3D &g, float v) { g.r       = v; });
        dn(color_sh_g,    [](Gaussian3D &g, float v) { g.g       = v; });
        dn(color_sh_b,    [](Gaussian3D &g, float v) { g.b       = v; });
        dn(logit_opacity, [](Gaussian3D &g, float v) { g.opacity = v; });

        if (sh_num_bands > 0) {
            for (int b = 0; b < sh_num_bands; b++) {
                CUDA_CHECK(cudaMemcpy(tmp.data(), sh_rest_r.ptr + b * n, n * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n; i++) host[i].sh_rest[b] = tmp[i];

                CUDA_CHECK(cudaMemcpy(tmp.data(), sh_rest_g.ptr + b * n, n * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n; i++) host[i].sh_rest[sh_num_bands + b] = tmp[i];

                CUDA_CHECK(cudaMemcpy(tmp.data(), sh_rest_b.ptr + b * n, n * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n; i++) host[i].sh_rest[2 * sh_num_bands + b] = tmp[i];
            }
        }

        return host;
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

    CudaBuffer<float> grad_sh_rest_r, grad_sh_rest_g, grad_sh_rest_b;
    int sh_num_bands = 0;

    void allocate(int n, int sh_degree = 0)
    {
        grad_pos_x.allocate(n);   grad_pos_y.allocate(n);   grad_pos_z.allocate(n);
        grad_scale_x.allocate(n); grad_scale_y.allocate(n); grad_scale_z.allocate(n);
        grad_rot_w.allocate(n);   grad_rot_x.allocate(n);
        grad_rot_y.allocate(n);   grad_rot_z.allocate(n);
        grad_color_sh_r.allocate(n);
        grad_color_sh_g.allocate(n);
        grad_color_sh_b.allocate(n);
        grad_logit_opacity.allocate(n);

        sh_num_bands = sh_degree_to_bands(sh_degree);
        if (sh_num_bands > 0) {
            grad_sh_rest_r.allocate(n * sh_num_bands);
            grad_sh_rest_g.allocate(n * sh_num_bands);
            grad_sh_rest_b.allocate(n * sh_num_bands);
        }
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
        if (sh_num_bands > 0) {
            grad_sh_rest_r.zero();
            grad_sh_rest_g.zero();
            grad_sh_rest_b.zero();
        }
    }
};