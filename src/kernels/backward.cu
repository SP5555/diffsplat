#include <cuda_runtime.h>
#include <math.h>

#include "backward.cuh"

#define ALPHA_THRESHOLD (1.0f / 255.0f)

/**
 * @brief Backward pass kernel to compute gradients for Gaussian splat parameters.
 * 
 * - N = number of splats
 * 
 * - M = number of [Key, Value] pairs.
 * 
 * Key is [tile ID | depth] and Value is splat ID.
 * M is approximately N but can be larger due to same splat in multiple tiles.
 * 
 * @param[in] pos_x             Splat x positions [N]
 * @param[in] pos_y             Splat y positions [N]
 * @param[in] cov_a             Splat covariance cxx [N]
 * @param[in] cov_b             Splat covariance cxy [N]
 * @param[in] cov_d             Splat covariance cyy [N]
 * @param[in] color_r           Splat color R [N]
 * @param[in] color_g           Splat color G [N]
 * @param[in] color_b           Splat color B [N]
 * @param[in] opacity           Splat opacity [N]
 * @param[in] d_target_pixels   Target pixel colors [H*W*3]
 * @param[in] d_values_sorted   Sorted contributing splat indices
 *                              for each tile [M]
 * @param[in] d_tile_ranges     Tile start/end indices
 *                              in d_values_sorted [num_tiles]
 * @param[in] d_pixels          Rendered pixel colors [H*W*3]
 * @param[in] d_T_final         Final transmittance values [H*W]
 * @param[in] d_n_contrib       Number of contributing splats
 *                              for each pixel [H*W]
 * @param[out] grad_pos_x       [N]
 * @param[out] grad_pos_y       [N]
 * @param[out] grad_cov_a       [N]
 * @param[out] grad_cov_b       [N]
 * @param[out] grad_cov_d       [N]
 * @param[out] grad_color_r     [N]
 * @param[out] grad_color_g     [N]
 * @param[out] grad_color_b     [N]
 * @param[out] grad_opacity     [N]
 * @param[in] num_tiles_x       Number of tiles in x direction
 * @param[in] num_tiles_y       Number of tiles in y direction
 * @param[in] screen_width      Screen width in pixels
 * @param[in] screen_height     Screen height in pixels
 */
__global__ void backwardKernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ cov_a,
    const float* __restrict__ cov_b,
    const float* __restrict__ cov_d,
    const float* __restrict__ color_r,
    const float* __restrict__ color_g,
    const float* __restrict__ color_b,
    const float* __restrict__ opacity,
    const float* __restrict__ d_target_pixels,
    const uint32_t* __restrict__ d_values_sorted,
    const int2* __restrict__ d_tile_ranges,
    const float* __restrict__ d_pixels,
    const float* __restrict__ d_T_final,
    const int* __restrict__ d_n_contrib,
    float *grad_pos_x,
    float *grad_pos_y,
    float *grad_cov_a,
    float *grad_cov_b,
    float *grad_cov_d,
    float *grad_color_r,
    float *grad_color_g,
    float *grad_color_b,
    float *grad_opacity,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= screen_width || pixel_y >= screen_height) return;

    int pixel_idx = pixel_y * screen_width + pixel_x;

    // recover tile ID, tile range and pixel NDC
    int tile_x = (pixel_x * num_tiles_x) / screen_width;
    int tile_y = (pixel_y * num_tiles_y) / screen_height;
    int tile_id = tile_y * num_tiles_x + tile_x;
    int2 tile_range = d_tile_ranges[tile_id];
    int n_contrib = d_n_contrib[pixel_idx];
    float x_ndc = (2.f * pixel_x + 1.f) / screen_width - 1.f;
    float y_ndc = (2.f * pixel_y + 1.f) / screen_height - 1.f;

    // recover T_final and walk backwards through
    // the splatting compositing to compute gradients
    // T_final is the final transmittance after compositing all contributing splats
    // then traverse them in reverse order using division trick.

    // small preparation pass: collect the contributing splat indices in order
    const int MAX_CONTRIB = 128;
    uint32_t contrib_indices[MAX_CONTRIB];
    int contributed_splats = 0;
    
    for (int idx = tile_range.x;
        idx < tile_range.y &&
        contributed_splats < n_contrib &&
        contributed_splats < MAX_CONTRIB;
        idx++)
    {
        uint32_t splat_id = d_values_sorted[idx];
        
        float dx = x_ndc - pos_x[splat_id];
        float dy = y_ndc - pos_y[splat_id];
        float cxx = cov_a[splat_id];
        float cxy = cov_b[splat_id];
        float cyy = cov_d[splat_id];
        float det = cxx * cyy - cxy * cxy;
        if (det < 1e-16f) continue; // skip degenerate splats
        
        float inv_det = 1.f / det;
        float inv_cxx =  cyy * inv_det;
        float inv_cxy = -cxy * inv_det;
        float inv_cyy =  cxx * inv_det;
        float dist2 = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
        if (dist2 > 9.f) continue;

        float alpha = fminf(0.99f, opacity[splat_id] * expf(-0.5f * dist2));
        if (alpha < ALPHA_THRESHOLD) continue;
        
        contrib_indices[contributed_splats++] = splat_id;
    }

    // backward pass
    // MSE loss gradient w.r.t. pixel color
    float scale = 1.f / (screen_width * screen_height);
    float dL_dCr = scale * 2.f * (d_pixels[3 * pixel_idx + 0] - d_target_pixels[3 * pixel_idx + 0]);
    float dL_dCg = scale * 2.f * (d_pixels[3 * pixel_idx + 1] - d_target_pixels[3 * pixel_idx + 1]);
    float dL_dCb = scale * 2.f * (d_pixels[3 * pixel_idx + 2] - d_target_pixels[3 * pixel_idx + 2]);

    // start from T_final and recover T_i by dividing out (1 - alpha_i)
    // in reverse order, going from the last splat to the first.
    float T_after = d_T_final[pixel_idx];
    float C_back_r = 0.f;
    float C_back_g = 0.f;
    float C_back_b = 0.f;

    for (int i = contributed_splats - 1; i >= 0; i--)
    {
        uint32_t splat_id = contrib_indices[i];

        // recompute geometry for this splat (same as above)
        float dx = x_ndc - pos_x[splat_id];
        float dy = y_ndc - pos_y[splat_id];
        float cxx = cov_a[splat_id];
        float cxy = cov_b[splat_id];
        float cyy = cov_d[splat_id];
        float det = cxx * cyy - cxy * cxy;
        float inv_det = 1.f / det;
        float inv_cxx =  cyy * inv_det;
        float inv_cxy = -cxy * inv_det;
        float inv_cyy =  cxx * inv_det;
        float dist2 = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
        float alpha = fminf(0.99f, opacity[splat_id] * expf(-0.5f * dist2));

        // recover T_before = T_after / (1 - alpha)
        float T_before = T_after / fmaxf(1.f - alpha, 1e-6f);

        float cR = color_r[splat_id];
        float cG = color_g[splat_id];
        float cB = color_b[splat_id];

        // gradient w.r.t. color
        // dL/dC_i = dL/dC_out * alpha
        atomicAdd(&grad_color_r[splat_id], alpha * T_before * dL_dCr);
        atomicAdd(&grad_color_g[splat_id], alpha * T_before * dL_dCg);
        atomicAdd(&grad_color_b[splat_id], alpha * T_before * dL_dCb);

        // gradient w.r.t. opacity
        // dL/dalpha_i = dL/dC_out * (C_i - C_back) * T_before
        float dL_dalpha = T_before * (
            (cR - C_back_r) * dL_dCr +
            (cG - C_back_g) * dL_dCg +
            (cB - C_back_b) * dL_dCb
        );
        float g = expf(-0.5f * dist2);
        atomicAdd(&grad_opacity[splat_id], dL_dalpha * g);

        // gradient w.r.t. dist2
        // dL/ddist2 = dL/dalpha_i * dalpha/ddist2
        float raw_alpha = opacity[splat_id] * g;
        float dL_ddist2 = dL_dalpha * (-0.5f) * raw_alpha;
        
        // gradient w.r.t. position
        // ddist2/dpos_x_i = -2 * (dx * inv_cxx + dy * inv_cxy)
        // ddist2/dpos_y_i = -2 * (dx * inv_cxy + dy * inv_cyy)
        float px_per_ndc_x = 2.f / screen_width;
        float px_per_ndc_y = 2.f / screen_height;
        float ddist2_dpos_x = -2.f * (dx * inv_cxx + dy * inv_cxy);
        float ddist2_dpos_y = -2.f * (dx * inv_cxy + dy * inv_cyy);
        atomicAdd(&grad_pos_x[splat_id], dL_ddist2 * ddist2_dpos_x * px_per_ndc_x);
        atomicAdd(&grad_pos_y[splat_id], dL_ddist2 * ddist2_dpos_y * px_per_ndc_y);

        // gradient w.r.t. covariance
        float dx2 = dx * dx;
        float dy2 = dy * dy;
        float dxdy = dx * dy;
        float inv_det2 = inv_det * inv_det;

        float ddist2_dcxx = -(dx*cyy - dy*cxy) * (dx*cyy - dy*cxy) * inv_det2;
        float ddist2_dcyy = -(dy*cxx - dx*cxy) * (dy*cxx - dx*cxy) * inv_det2;
        float ddist2_dcxy = 2.f * (cxy*(dx2*cyy + dy2*cxx) - dxdy*(cxx*cyy + cxy*cxy)) * inv_det2;
        atomicAdd(&grad_cov_a[splat_id], dL_ddist2 * ddist2_dcxx);
        atomicAdd(&grad_cov_b[splat_id], dL_ddist2 * ddist2_dcxy);
        atomicAdd(&grad_cov_d[splat_id], dL_ddist2 * ddist2_dcyy);

        // update C_back for next iteration
        C_back_r += cR * alpha * T_before;
        C_back_g += cG * alpha * T_before;
        C_back_b += cB * alpha * T_before;

        // update T_after for next iteration
        // T_after is transmittance after compositing splat i
        // T_before is transmittance before compositing splat i
        // therefore, T_before becomes T_after for the next iteration
        T_after = T_before;
    }

    return;
}

void launchBackward(
    const GaussianParams &gaussians,
    const GaussianOptState &opt_state,
    const float *d_target_pixels,
    const uint32_t *d_values_sorted,
    const int2 *d_tile_ranges,
    const float *d_pixels,
    const float *d_T_final,
    const int *d_n_contrib,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (screen_width + threads.x - 1) / threads.x,
        (screen_height + threads.y - 1) / threads.y
    );
    backwardKernel<<<blocks, threads>>>(
        gaussians.pos_x,
        gaussians.pos_y,
        gaussians.cov_a,
        gaussians.cov_b,
        gaussians.cov_d,
        gaussians.color_r,
        gaussians.color_g,
        gaussians.color_b,
        gaussians.opacity,
        d_target_pixels,
        d_values_sorted,
        d_tile_ranges,
        d_pixels,
        d_T_final,
        d_n_contrib,
        opt_state.grad_pos_x,
        opt_state.grad_pos_y,
        opt_state.grad_cov_a,
        opt_state.grad_cov_b,
        opt_state.grad_cov_d,
        opt_state.grad_color_r,
        opt_state.grad_color_g,
        opt_state.grad_color_b,
        opt_state.grad_opacity,
        num_tiles_x,
        num_tiles_y,
        screen_width,
        screen_height
    );
    cudaDeviceSynchronize();
}