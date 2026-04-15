#include "gsplat_rasterize_layer.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

/* ===== ===== Constants ===== ===== */

// Tile size is 16x16 pixels with one CUDA thread per pixel.
// 16 is chosen so that TILE_PIXELS = 256, a multiple of the warp size (32)
// that fills 8 warps per block -- a good occupancy sweet spot on all NVIDIA
// architectures.  All shared-memory arrays and 1D kernel block sizes derive
// from TILE_SIZE so that changing it stays consistent.
static constexpr int TILE_SIZE   = 16;
static constexpr int TILE_PIXELS = TILE_SIZE * TILE_SIZE; // 256 = 8 warps

#define GSPLAT_MAX_ALPHA   0.99f
#define GSPLAT_ALPHA_THRES (1.0f / 255.0f)
#define GSPLAT_T_THRES     0.0001f

/* ===== ===== Tile Assign ===== ===== */

__device__ inline uint64_t makeKey(uint32_t tile_id, float depth)
{
    uint32_t u = __float_as_uint(depth);
    if (u >> 31)
        u = ~u;
    else
        u ^= 0x80000000u;
    return ((uint64_t)tile_id << 32) | u;
}

/**
 * Tile assignment following gsplat's convention exactly.
 *
 * Tile boundaries are derived from pixel coordinates, not NDC fractions:
 *   tile_x = floor(pixel_x / TILE_SIZE)
 *           = floor((ndc_x + 1) / 2 * screen_width / TILE_SIZE)
 *
 * This ensures tile boundaries align precisely with the 16x16 pixel blocks
 * used by the rasterize kernel, regardless of whether the resolution is a
 * multiple of TILE_SIZE.  Passing an external num_tiles_x would misalign
 * tile boundaries whenever (num_tiles_x * TILE_SIZE != screen_width).
 */
__global__ void gsTileAssignKernel(
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_z,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    int splat_count,
    uint64_t *isect_ids,
    uint32_t *gauss_ids,
    uint32_t *n_isects,
    uint32_t *visible_count,
    int max_isects,
    int num_tiles_x,
    int num_tiles_y,
    int screen_width,
    int screen_height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= splat_count) return;

    float x = ndc_x[i];
    float y = ndc_y[i];
    float z = ndc_z[i];

    if (fabsf(x) > 1.0f || fabsf(y) > 1.0f || fabsf(z) > 1.0f)
        return;

    float cxx = ndc_cxx[i];
    float cxy = ndc_cxy[i];
    float cyy = ndc_cyy[i];

    float det = cxx * cyy - cxy * cxy;
    if (!(det > 0.f)) return;

    float trace   = cxx + cyy;
    float temp    = fmaxf(0.f, trace * trace - 4.f * det);
    float lambda2 = 0.5f * (trace + sqrtf(temp));

    float pixel_ndc = fmaxf(2.f / screen_width, 2.f / screen_height);
    if (3.f * sqrtf(lambda2 * 2.f) < pixel_ndc) return;

    float lambda1    = 0.5f * (trace - sqrtf(temp));
    float max_radius = 3.f * sqrtf(fmaxf(lambda1, lambda2));

    float min_x = x - max_radius, max_x = x + max_radius;
    float min_y = y - max_radius, max_y = y + max_radius;

    // Convert NDC bounding box to tile indices via pixel coordinates.
    // ndc -> pixel: px = (ndc + 1) / 2 * W,  py = (1 - ndc) / 2 * H
    // pixel -> tile: tx = floor(px / TILE_SIZE)
    // Combined:     tx = floor((ndc + 1) / 2 * W / TILE_SIZE)
    const float fx = (float)screen_width  / TILE_SIZE;
    const float fy = (float)screen_height / TILE_SIZE;
    auto ndcToTileX = [&](float v) { return (int)floorf((v + 1.f) * 0.5f * fx); };
    auto ndcToTileY = [&](float v) { return (int)floorf((1.f - v) * 0.5f * fy); };

    int tx0 = max(ndcToTileX(min_x), 0), tx1 = min(ndcToTileX(max_x), num_tiles_x - 1);
    int ty0 = max(ndcToTileY(max_y), 0), ty1 = min(ndcToTileY(min_y), num_tiles_y - 1);
    if (tx0 > tx1 || ty0 > ty1) return;

    atomicAdd(visible_count, 1u);
    for (int ty = ty0; ty <= ty1; ty++)
    for (int tx = tx0; tx <= tx1; tx++)
    {
        uint32_t tile_id = (uint32_t)(ty * num_tiles_x + tx);
        uint64_t key     = makeKey(tile_id, z);
        uint32_t slot    = atomicAdd(n_isects, 1u);
        if (slot >= (uint32_t)max_isects) { atomicSub(n_isects, 1u); return; }
        isect_ids[slot] = key;
        gauss_ids[slot] = (uint32_t)i;
    }
}

/* ===== ===== Build Tile Ranges (identical to RasterizeLayer) ===== ===== */

__global__ void gsBuildTileRangesKernel(
    const uint64_t *keys_sorted,
    int2 *tile_offsets,
    uint32_t n_isects,
    int num_tiles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint32_t)i >= n_isects) return;
    int tile_id = (int)(keys_sorted[i] >> 32);
    if (tile_id < 0 || tile_id >= num_tiles) return;
    if (i == 0 || (int)(keys_sorted[i - 1] >> 32) != tile_id)
        tile_offsets[tile_id].x = i;
    if (i == (int)n_isects - 1 || (int)(keys_sorted[i + 1] >> 32) != tile_id)
        tile_offsets[tile_id].y = i + 1;
}

/* ===== ===== Warp Utilities ===== ===== */

// Warp-level sum reduction: lane 0 gets the sum. Sufficient for the atomicAdd
// pattern where only lane 0 writes.
__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

// Warp-level max all-reduce: every lane gets the same maximum value.
// Used to compute warp_bin_final for the batch-skip optimization in the
// backward pass.
__device__ __forceinline__ int warpAllReduceMax(int val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffffu, val, mask));
    return val;
}

/* ===== ===== Forward Kernel ===== ===== */

/**
 * Adapted from gsplat RasterizeToPixels3DGSFwd.cu.
 *
 * One 16x16 CUDA block per tile; one thread per pixel. Gaussians are
 * streamed front-to-back in chunks of TILE_PIXELS (=256), loaded
 * cooperatively into shared memory. Alpha compositing terminates early
 * per-pixel via T < T_THRES.
 *
 * Inputs use the same NDC + SoA layout as Splat2DParams (no AoS conversion).
 * Covariance inverse is computed during the shared-memory load, not stored
 * separately.
 *
 * Saves last_ids (sorted-list index of the last contributing Gaussian per
 * pixel) for the backward pass - replaces the n_contrib approach.
 */
__global__ void gsplatFwdKernel(
    const float    *__restrict__ ndc_x,
    const float    *__restrict__ ndc_y,
    const float    *__restrict__ cov_xx,
    const float    *__restrict__ cov_xy,
    const float    *__restrict__ cov_yy,
    const float    *__restrict__ color_r,
    const float    *__restrict__ color_g,
    const float    *__restrict__ color_b,
    const float    *__restrict__ opacity,
    const uint32_t *__restrict__ flatten_ids, // values_sorted
    const int2     *__restrict__ tile_offsets, // {start, end} per tile
    float    *__restrict__ render_colors,      // [H*W*3]
    float    *__restrict__ render_alphas,      // [H*W]
    int32_t  *__restrict__ last_ids,           // [H*W]
    int image_width,
    int image_height,
    int num_tiles_x)
{
    __shared__ float sh_x   [TILE_PIXELS];
    __shared__ float sh_y   [TILE_PIXELS];
    __shared__ float sh_icxx[TILE_PIXELS]; // cyy / det
    __shared__ float sh_icxy[TILE_PIXELS]; // -cxy / det
    __shared__ float sh_icyy[TILE_PIXELS]; // cxx / det
    __shared__ float sh_r   [TILE_PIXELS];
    __shared__ float sh_g   [TILE_PIXELS];
    __shared__ float sh_b   [TILE_PIXELS];
    __shared__ float sh_a   [TILE_PIXELS];

    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;
    int tile_id  = tile_row * num_tiles_x + tile_col;

    int i = tile_row * TILE_SIZE + threadIdx.y; // pixel row
    int j = tile_col * TILE_SIZE + threadIdx.x; // pixel col

    // NDC coordinate of this pixel's centre.
    // Matches the convention in tileAssignKernel and Splat2DParams.
    float px_ndc = 2.f * (j + 0.5f) / image_width  - 1.f;
    float py_ndc = 1.f - 2.f * (i + 0.5f) / image_height;

    int  pix_id = i * image_width + j;
    bool inside = (i < image_height && j < image_width);
    bool done   = !inside;

    int2     range      = tile_offsets[tile_id];
    int      range_start = range.x;
    int      range_end   = range.y;
    uint32_t num_batches = ((uint32_t)(range_end - range_start) + TILE_PIXELS - 1) / TILE_PIXELS;

    uint32_t tr = threadIdx.y * TILE_SIZE + threadIdx.x; // thread rank in block

    float T     = 1.f;
    uint32_t cur_idx = 0;
    float C_r = 0.f, C_g = 0.f, C_b = 0.f;

    for (uint32_t b = 0; b < num_batches; ++b)
    {
        // Early exit if every pixel in the tile is done.
        if (__syncthreads_count(done) >= TILE_PIXELS)
            break;

        // Cooperative load of one batch from front to back.
        uint32_t batch_start = (uint32_t)range_start + (uint32_t)TILE_PIXELS * b;
        uint32_t idx         = batch_start + tr;
        if (idx < (uint32_t)range_end)
        {
            uint32_t g  = flatten_ids[idx];
            float cxx   = cov_xx[g], cxy = cov_xy[g], cyy = cov_yy[g];
            float det   = cxx * cyy - cxy * cxy;
            // tileAssignKernel already filtered det <= 0, so det > 0 here.
            float idet  = 1.f / det;
            sh_x   [tr] = ndc_x  [g];
            sh_y   [tr] = ndc_y  [g];
            sh_icxx[tr] =  cyy * idet;
            sh_icxy[tr] = -cxy * idet;
            sh_icyy[tr] =  cxx * idet;
            sh_r   [tr] = color_r[g];
            sh_g   [tr] = color_g[g];
            sh_b   [tr] = color_b[g];
            sh_a   [tr] = opacity [g];
        }
        __syncthreads();

        uint32_t batch_size = min((uint32_t)TILE_PIXELS, (uint32_t)range_end - batch_start);
        for (uint32_t t = 0; t < batch_size && !done; ++t)
        {
            float dx    = sh_x[t] - px_ndc;
            float dy    = sh_y[t] - py_ndc;
            float dist2 = dx*dx*sh_icxx[t] + 2.f*dx*dy*sh_icxy[t] + dy*dy*sh_icyy[t];
            float sigma = 0.5f * dist2;
            if (sigma < 0.f) continue;

            float alpha = fminf(GSPLAT_MAX_ALPHA, sh_a[t] * __expf(-sigma));
            if (alpha < GSPLAT_ALPHA_THRES) continue;

            // Accumulate colour and update transmittance before the saturation check
            float vis = alpha * T;
            C_r     += sh_r[t] * vis;
            C_g     += sh_g[t] * vis;
            C_b     += sh_b[t] * vis;
            cur_idx  = batch_start + t;
            T        = T * (1.f - alpha);

            if (T <= GSPLAT_T_THRES) done = true;
        }
        __syncthreads();
    }

    if (inside)
    {
        render_colors[pix_id * 3 + 0] = C_r;
        render_colors[pix_id * 3 + 1] = C_g;
        render_colors[pix_id * 3 + 2] = C_b;
        render_alphas[pix_id]          = 1.f - T;
        last_ids     [pix_id]          = (int32_t)cur_idx;
    }
}

/* ===== ===== Backward Kernel ===== ===== */

/**
 * Adapted from gsplat RasterizeToPixels3DGSBwd.cu.
 *
 * Key improvements over RasterizeLayer's backwardKernel:
 *
 *  1. Warp-level gradient reduction (warpReduceSum via __shfl_down_sync):
 *     All 32 lanes sum their local gradient for each Gaussian, then only
 *     lane 0 does the global atomicAdd. Reduces atomic pressure ~32x.
 *
 *  2. last_ids-based traversal:
 *     Uses the sorted-list index of each pixel's last contributing Gaussian
 *     (saved in the forward) instead of a contrib_pos[] stack. No stack
 *     overflow risk, no MAX_CONTRIB limit.
 *
 *  3. Warp-level batch skipping:
 *     warp_bin_final = max(last_ids) across the warp. Gaussians with sorted
 *     index > warp_bin_final are skipped entirely (no thread in the warp
 *     needs them), saving the inner loop for early-terminated pixels.
 *
 * Gradient formulas use the same NDC-space covariance identities as
 * RasterizeLayer::backwardKernel (reformulated to avoid storing raw cxx/cyy):
 *   ddist2/dcxx = -(dx.icyy - dy.icxy)^2 . det   = -1/4.ddist2_dx^2
 *   ddist2/dcyy = -(dy.icxx - dx.icxy)^2 . det   = -1/4.ddist2_dy^2
 *   ddist2/dcxy = -2.icxy.dist2 - 2.dx.dy.idet
 */
__global__ void gsplatBwdKernel(
    // forward inputs
    const float    *__restrict__ ndc_x,
    const float    *__restrict__ ndc_y,
    const float    *__restrict__ cov_xx,
    const float    *__restrict__ cov_xy,
    const float    *__restrict__ cov_yy,
    const float    *__restrict__ color_r,
    const float    *__restrict__ color_g,
    const float    *__restrict__ color_b,
    const float    *__restrict__ opacity,
    const uint32_t *__restrict__ flatten_ids,
    const int2     *__restrict__ tile_offsets,
    // forward outputs (saved)
    const float    *__restrict__ render_alphas,
    const int32_t  *__restrict__ last_ids,
    // incoming gradient
    const float    *__restrict__ v_render_colors, // dL/d_render_colors [H*W*3]
    // gradient outputs (accumulated)
    float *v_ndc_x,
    float *v_ndc_y,
    float *v_cov_xx,
    float *v_cov_xy,
    float *v_cov_yy,
    float *v_color_r,
    float *v_color_g,
    float *v_color_b,
    float *v_opacities,
    int image_width,
    int image_height,
    int num_tiles_x)
{
    __shared__ uint32_t sh_id  [TILE_PIXELS];
    __shared__ float    sh_x   [TILE_PIXELS];
    __shared__ float    sh_y   [TILE_PIXELS];
    __shared__ float    sh_icxx[TILE_PIXELS];
    __shared__ float    sh_icxy[TILE_PIXELS];
    __shared__ float    sh_icyy[TILE_PIXELS];
    __shared__ float    sh_idet[TILE_PIXELS];
    __shared__ float    sh_r   [TILE_PIXELS];
    __shared__ float    sh_g   [TILE_PIXELS];
    __shared__ float    sh_b   [TILE_PIXELS];
    __shared__ float    sh_a   [TILE_PIXELS];

    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;
    int tile_id  = tile_row * num_tiles_x + tile_col;

    int i = tile_row * TILE_SIZE + threadIdx.y;
    int j = tile_col * TILE_SIZE + threadIdx.x;

    float px_ndc = 2.f * (j + 0.5f) / image_width  - 1.f;
    float py_ndc = 1.f - 2.f * (i + 0.5f) / image_height;

    bool inside = (i < image_height && j < image_width);
    // Clamp pix_id for out-of-bounds threads; `inside` guards all output writes.
    int pix_id = min(i * image_width + j, image_width * image_height - 1);

    int2 range      = tile_offsets[tile_id];
    int  range_start = range.x;
    int  range_end   = range.y;
    if (range_end <= range_start) return;

    uint32_t num_batches = ((uint32_t)(range_end - range_start) + TILE_PIXELS - 1) / TILE_PIXELS;

    uint32_t tr        = threadIdx.y * TILE_SIZE + threadIdx.x;
    uint32_t warp_lane = tr % 32;

    // Per-pixel state (T is recovered from T_final going back-to-front).
    float T_final = 1.f - render_alphas[pix_id];
    float T       = T_final;
    float accum_r = 0.f, accum_g = 0.f, accum_b = 0.f; // colour of Gaussians farther than current

    // Index in the sorted list of the last Gaussian that contributed to this pixel.
    // -1 for out-of-bounds threads so they don't affect warp_bin_final.
    int32_t bin_final     = inside ? last_ids[pix_id] : -1;
    int32_t warp_bin_final = warpAllReduceMax(bin_final);

    float v_render_r = inside ? v_render_colors[pix_id * 3 + 0] : 0.f;
    float v_render_g = inside ? v_render_colors[pix_id * 3 + 1] : 0.f;
    float v_render_b = inside ? v_render_colors[pix_id * 3 + 2] : 0.f;

    // Process batches back-to-front (b=0 covers the farthest Gaussians).
    for (uint32_t b = 0; b < num_batches; ++b)
    {
        __syncthreads(); // before each shared-mem load

        // batch_end: absolute sorted-list index of the farthest Gaussian in this batch.
        int batch_end  = range_end - 1 - (int)TILE_PIXELS * (int)b;
        int batch_size = min(TILE_PIXELS, batch_end + 1 - range_start);

        // Cooperative load (back-to-front within batch: tr=0 gets batch_end).
        int load_idx = batch_end - (int)tr;
        if (load_idx >= range_start && load_idx < range_end)
        {
            uint32_t g  = flatten_ids[load_idx];
            float cxx   = cov_xx[g], cxy = cov_xy[g], cyy = cov_yy[g];
            float det   = cxx * cyy - cxy * cxy;
            float idet  = 1.f / det;
            sh_id  [tr] = g;
            sh_x   [tr] = ndc_x  [g];
            sh_y   [tr] = ndc_y  [g];
            sh_icxx[tr] =  cyy * idet;
            sh_icxy[tr] = -cxy * idet;
            sh_icyy[tr] =  cxx * idet;
            sh_idet[tr] = idet;
            sh_r   [tr] = color_r[g];
            sh_g   [tr] = color_g[g];
            sh_b   [tr] = color_b[g];
            sh_a   [tr] = opacity [g];
        }
        __syncthreads();

        // Skip Gaussians no thread in this warp contributed to.
        // t=0 -> Gaussian at batch_end (farthest); t increases toward front.
        int t_start = max(0, batch_end - warp_bin_final);

        for (int t = t_start; t < batch_size; ++t)
        {
            // This thread's pixel needs Gaussian at (batch_end - t) only if it
            // falls within [range_start, bin_final].
            bool valid = inside && ((batch_end - t) <= (int)bin_final);

            float alpha = 0.f, opac = 0.f, g_exp = 0.f;
            float dx = 0.f, dy = 0.f, dist2 = 0.f, idet_t = 0.f;

            if (valid)
            {
                dx    = sh_x[t] - px_ndc;
                dy    = sh_y[t] - py_ndc;
                idet_t = sh_idet[t];
                dist2 = dx*dx*sh_icxx[t] + 2.f*dx*dy*sh_icxy[t] + dy*dy*sh_icyy[t];
                float sigma = 0.5f * dist2;
                if (sigma < 0.f) { valid = false; }
                else {
                    opac  = sh_a[t];
                    g_exp = __expf(-sigma);
                    alpha = fminf(GSPLAT_MAX_ALPHA, opac * g_exp);
                    if (alpha < GSPLAT_ALPHA_THRES) valid = false;
                }
            }

            // Skip if no thread in the warp is valid for this Gaussian.
            if (!__any_sync(0xffffffffu, valid)) continue;

            float v_r_l   = 0.f, v_g_l   = 0.f, v_b_l   = 0.f;
            float v_x_l   = 0.f, v_y_l   = 0.f;
            float v_cxx_l = 0.f, v_cxy_l = 0.f, v_cyy_l = 0.f;
            float v_a_l   = 0.f;

            if (valid)
            {
                // Recover T_before from T_after using the compositing identity.
                float T_before = T / fmaxf(1.f - alpha, 1e-6f);
                float fac      = alpha * T_before;

                // dL / d(color)
                v_r_l = fac * v_render_r;
                v_g_l = fac * v_render_g;
                v_b_l = fac * v_render_b;

                // dL / d(alpha)
                float v_alpha =
                    T_before * ((sh_r[t] - accum_r) * v_render_r +
                                (sh_g[t] - accum_g) * v_render_g +
                                (sh_b[t] - accum_b) * v_render_b);

                // dL / d(sigma) = -opac . exp(-sigma) . v_alpha
                // (chain through alpha = min(MAX_ALPHA, opac.exp(-sigma)))
                if (opac * g_exp <= GSPLAT_MAX_ALPHA)
                {
                    v_a_l = g_exp * v_alpha;           // dL/d(opacity)
                    float v_sigma = -opac * g_exp * v_alpha;
                    float v_dist2 = 0.5f * v_sigma;   // dL/d(dist2)

                    // dL/d(ndc_x[g]) and dL/d(ndc_y[g])
                    float ddist2_dx = 2.f * (dx*sh_icxx[t] + dy*sh_icxy[t]);
                    float ddist2_dy = 2.f * (dx*sh_icxy[t] + dy*sh_icyy[t]);
                    v_x_l = v_dist2 * ddist2_dx;
                    v_y_l = v_dist2 * ddist2_dy;

                    // dL/d(cov_xx/xy/yy) via reformulated chain rule
                    // (avoids storing inv_cov in forward; see header comment)
                    v_cxx_l = v_dist2 * (-0.25f * ddist2_dx * ddist2_dx);
                    v_cyy_l = v_dist2 * (-0.25f * ddist2_dy * ddist2_dy);
                    v_cxy_l = v_dist2 * (-2.f * sh_icxy[t] * dist2 - 2.f * dx * dy * idet_t);
                }

                // Update running accumulators for next (closer) Gaussian.
                accum_r += sh_r[t] * fac;
                accum_g += sh_g[t] * fac;
                accum_b += sh_b[t] * fac;
                T      = T_before;
            }

            // Warp-level reduction: accumulate across 32 lanes, then lane 0
            // writes a single global atomicAdd per gradient.
            v_r_l   = warpReduceSum(v_r_l);
            v_g_l   = warpReduceSum(v_g_l);
            v_b_l   = warpReduceSum(v_b_l);
            v_x_l   = warpReduceSum(v_x_l);
            v_y_l   = warpReduceSum(v_y_l);
            v_cxx_l = warpReduceSum(v_cxx_l);
            v_cxy_l = warpReduceSum(v_cxy_l);
            v_cyy_l = warpReduceSum(v_cyy_l);
            v_a_l   = warpReduceSum(v_a_l);

            if (warp_lane == 0)
            {
                uint32_t g = sh_id[t];
                atomicAdd(&v_color_r[g], v_r_l);
                atomicAdd(&v_color_g[g], v_g_l);
                atomicAdd(&v_color_b[g], v_b_l);
                atomicAdd(&v_ndc_x  [g], v_x_l);
                atomicAdd(&v_ndc_y  [g], v_y_l);
                atomicAdd(&v_cov_xx [g], v_cxx_l);
                atomicAdd(&v_cov_xy [g], v_cxy_l);
                atomicAdd(&v_cov_yy [g], v_cyy_l);
                atomicAdd(&v_opacities[g], v_a_l);
            }
        }
    }
}

/* ===== ===== Utils ===== ===== */

uint32_t GsplatRasterizeLayer::getVisibleCount()
{
    uint32_t c = 0;
    CUDA_CHECK(cudaMemcpy(&c, d_visible_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return c;
}

/* ===== ===== Lifecycle ===== ===== */

void GsplatRasterizeLayer::allocate(int width, int height, int count)
{
    screen_width  = width;
    screen_height = height;
    num_pixels    = width * height;
    // Tile counts are always derived from the resolution and TILE_SIZE so that
    // tile boundaries align exactly with the 16x16 pixel blocks in the kernels.
    num_tiles_x   = (width  + TILE_SIZE - 1) / TILE_SIZE;
    num_tiles_y   = (height + TILE_SIZE - 1) / TILE_SIZE;
    // The tile-assign kernel degrades gracefully (drops excess pairs) if exceeded.
    max_isects     = (1 << 25);
    int numTiles  = num_tiles_x * num_tiles_y;

    d_render_colors.allocate  (num_pixels * 3);
    d_render_alphas.allocate(num_pixels);
    d_last_ids.allocate    (num_pixels);

    d_isect_ids.allocate         (max_isects);
    d_gauss_ids.allocate       (max_isects);
    d_isect_ids_sorted.allocate  (max_isects);
    d_flatten_ids.allocate(max_isects);
    d_n_isects.allocate   (1);
    d_visible_count.allocate(1);
    d_tile_offsets.allocate  (numTiles);

    {
        size_t required = 0;
        uint64_t *dummy_keys = nullptr;
        uint32_t *dummy_vals = nullptr;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, required,
            dummy_keys, dummy_keys,
            dummy_vals, dummy_vals,
            max_isects));
        d_sort_temp.allocate(required);
        sort_temp_bytes = required;
    }
}

void GsplatRasterizeLayer::allocateGrad(int count)
{
    grad_in.allocate(count);
}

void GsplatRasterizeLayer::zero_grad()
{
    grad_in.zero_grad();
}

void GsplatRasterizeLayer::resize(int new_width, int new_height)
{
    if (new_width == screen_width && new_height == screen_height)
        return;
    screen_width  = new_width;
    screen_height = new_height;
    num_pixels    = screen_width * screen_height;
    num_tiles_x   = (screen_width  + TILE_SIZE - 1) / TILE_SIZE;
    num_tiles_y   = (screen_height + TILE_SIZE - 1) / TILE_SIZE;
    d_render_colors.allocate  (num_pixels * 3);
    d_render_alphas.allocate(num_pixels);
    d_last_ids.allocate    (num_pixels);
    d_tile_offsets.allocate (num_tiles_x * num_tiles_y);
}

/* ===== ===== Forward / Backward ===== ===== */

void GsplatRasterizeLayer::forward()
{
    int numTiles = num_tiles_x * num_tiles_y;

    CUDA_CHECK(cudaMemset(d_render_colors,    0, num_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_n_isects,    0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_visible_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_tile_offsets,   0, numTiles * sizeof(int2)));

    // Tile assign
    {
        int blocks = (in->count + TILE_PIXELS - 1) / TILE_PIXELS;
        gsTileAssignKernel<<<blocks, TILE_PIXELS>>>(
            in->pos_x, in->pos_y, in->pos_z,
            in->cov_xx, in->cov_xy, in->cov_yy,
            in->count,
            d_isect_ids, d_gauss_ids, d_n_isects,
            d_visible_count,
            max_isects, num_tiles_x, num_tiles_y,
            screen_width, screen_height);
        CUDA_SYNC_CHECK();
    }

    uint32_t n_isects = 0;
    CUDA_CHECK(cudaMemcpy(&n_isects, d_n_isects, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (n_isects == 0) return;

    // Radix sort (tile_id | depth, splat_id)
    {
        size_t required = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, required,
            d_isect_ids.ptr, d_isect_ids_sorted.ptr,
            d_gauss_ids.ptr, d_flatten_ids.ptr,
            (int)n_isects));

        if (required > sort_temp_bytes)
        {
            d_sort_temp.allocate(required);
            sort_temp_bytes = required;
        }

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            (void *)d_sort_temp.ptr, required,
            d_isect_ids.ptr, d_isect_ids_sorted.ptr,
            d_gauss_ids.ptr, d_flatten_ids.ptr,
            (int)n_isects));
    }

    // Build per-tile {start, end} ranges in the sorted list
    {
        int blocks = ((int)n_isects + TILE_PIXELS - 1) / TILE_PIXELS;
        gsBuildTileRangesKernel<<<blocks, TILE_PIXELS>>>(
            d_isect_ids_sorted, d_tile_offsets, n_isects, numTiles);
        CUDA_SYNC_CHECK();
    }

    // Rasterize: one 16x16 block per tile
    {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 grid(num_tiles_y, num_tiles_x);
        gsplatFwdKernel<<<grid, threads>>>(
            in->pos_x, in->pos_y,
            in->cov_xx, in->cov_xy, in->cov_yy,
            in->color_r, in->color_g, in->color_b,
            in->opacity,
            d_flatten_ids, d_tile_offsets,
            d_render_colors, d_render_alphas, d_last_ids,
            screen_width, screen_height, num_tiles_x);
        CUDA_SYNC_CHECK();
    }
}

void GsplatRasterizeLayer::backward()
{
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid(num_tiles_y, num_tiles_x);
    gsplatBwdKernel<<<grid, threads>>>(
        in->pos_x, in->pos_y,
        in->cov_xx, in->cov_xy, in->cov_yy,
        in->color_r, in->color_g, in->color_b,
        in->opacity,
        d_flatten_ids, d_tile_offsets,
        d_render_alphas, d_last_ids,
        v_render_colors,
        grad_in.grad_pos_x,   grad_in.grad_pos_y,
        grad_in.grad_cov_xx,  grad_in.grad_cov_xy,  grad_in.grad_cov_yy,
        grad_in.grad_color_r, grad_in.grad_color_g, grad_in.grad_color_b,
        grad_in.grad_opacity,
        screen_width, screen_height, num_tiles_x);
    CUDA_SYNC_CHECK();
}
