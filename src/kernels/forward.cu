#include "forward.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define T_THRESHOLD 0.0001f
#define ALPHA_THRESHOLD (1.0f / 255.0f)

/**
 * @brief Forward rasterization kernel
 * 
 * Iterates over sorted splats per tile, alpha compositing front-to-back.
 * Stores final pixel colors and T values for backward pass.
 * 
 * @param pos_x             Splat x positions in NDC
 * @param pos_y             Splat y positions in NDC
 * @param cov_a             Splat covariance matrix element cxx
 * @param cov_b             Splat covariance matrix element cxy
 * @param cov_d             Splat covariance matrix element cyy
 * @param color_r           Splat color R
 * @param color_g           Splat color G
 * @param color_b           Splat color B
 * @param opacity           Splat opacity
 * @param values_sorted     Sorted splat IDs on the device, sorted by tile_id and depth
 * @param tile_ranges       Tile ranges for each tile
 * @param d_pixels          Output pixel colors (RGB)
 * @param d_T_final         Final T values for backward pass
 * @param d_n_contrib       Number of contributing splats per pixel
 * @param num_tiles_x       Number of tiles in x direction
 * @param num_tiles_y       Number of tiles in y direction
 * @param screen_width      Screen width in pixels
 * @param screen_height     Screen height in pixels
 */
__global__ void forwardKernel(
    const float *__restrict__ pos_x,
    const float *__restrict__ pos_y,
    const float *__restrict__ cov_a,
    const float *__restrict__ cov_b,
    const float *__restrict__ cov_d,
    const float *__restrict__ color_r,
    const float *__restrict__ color_g,
    const float *__restrict__ color_b,
    const float *__restrict__ opacity,
    const uint32_t *__restrict__ values_sorted,
    const int2 *__restrict__ tile_ranges,
    float *d_pixels,
    float *d_T_final,
    int *d_n_contrib,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_x = (pixel_x * num_tiles_x) / screen_width;
    int tile_y = (pixel_y * num_tiles_y) / screen_height;
    int tile_id = tile_y * num_tiles_x + tile_x;
    int2 range = tile_ranges[tile_id];

    bool inside = (pixel_x < screen_width && pixel_y < screen_height);

    if (!inside) {
        return;
    }

    float ndc_x = (2.0f * (pixel_x + 0.5f) / (float)screen_width) - 1.0f;
    float ndc_y = (2.0f * (pixel_y + 0.5f) / (float)screen_height) - 1.0f;

    // colors
    float C_r = 0.f, C_g = 0.f, C_b = 0.f;
    // transmittance
    float T = 1.0f;
    int contrib = 0;

    for (int idx = range.x; idx < range.y; idx++)
    {
        uint32_t sid = values_sorted[idx];

        float dx = ndc_x - pos_x[sid];
        float dy = ndc_y - pos_y[sid];

        float cxx = cov_a[sid];
        float cxy = cov_b[sid];
        float cyy = cov_d[sid];
        float det = cxx * cyy - cxy * cxy;
        if (det <= 0.f)
            continue;

        float inv_det = 1.0f / det;
        float inv_cxx = cyy * inv_det;
        float inv_cxy = -cxy * inv_det;
        float inv_cyy = cxx * inv_det;

        // maahalanobis distance squared
        float dist2 = dx * dx * inv_cxx + 2.0f * dx * dy * inv_cxy + dy * dy * inv_cyy;
        if (dist2 > 9.0f)
            continue;

        float alpha = fminf(0.99f, opacity[sid] * expf(-0.5f * dist2));
        if (alpha < ALPHA_THRESHOLD)
            continue;

        C_r += color_r[sid] * alpha * T;
        C_g += color_g[sid] * alpha * T;
        C_b += color_b[sid] * alpha * T;
        T *= (1.0f - alpha);
        contrib++;

        if (T < T_THRESHOLD)
            break;
    }

    int pidx = pixel_y * screen_width + pixel_x;
    d_pixels[pidx * 3 + 0] = C_r;
    d_pixels[pidx * 3 + 1] = C_g;
    d_pixels[pidx * 3 + 2] = C_b;
    d_T_final[pidx] = T;
    d_n_contrib[pidx] = contrib;
}

void launchForward(
    const GaussianParams &gaussians,
    const uint32_t *d_values_sorted,
    const int2 *d_tile_ranges,
    float *d_pixels,
    float *d_T_final,
    int *d_n_contrib,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    (void)num_tiles_y;

    dim3 threads(16, 16);
    dim3 blocks(
        (screen_width + threads.x - 1) / threads.x,
        (screen_height + threads.y - 1) / threads.y
    );

    forwardKernel<<<blocks, threads>>>(
        gaussians.pos_x, gaussians.pos_y,
        gaussians.cov_a, gaussians.cov_b, gaussians.cov_d,
        gaussians.color_r, gaussians.color_g, gaussians.color_b,
        gaussians.opacity,
        d_values_sorted, d_tile_ranges,
        d_pixels, d_T_final, d_n_contrib,
        num_tiles_x, num_tiles_y,
        screen_width, screen_height
    );
}
