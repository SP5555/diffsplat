#include "tile_assign.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief Creates a 64-bit key tile_id and depth for radix sorting.
 *
 * Dudes from IEEE guarantee that reinterpreting a positive float as uint
 * preserves sort order, so radix sort on the key = sort by tile first,
 * then by depth.
 *
 * @param[in] tile_id   ID of the tile this splat overlaps with
 *                      (0-based, row-major)
 * @param[in] depth     Depth of the splat
 *                      (Z in NDC [-1, 1] with -1 near and 1 far)
 * @return              64-bit key for radix sorting
 */
__device__ inline uint64_t makeKey(uint32_t tile_id, float depth)
{
    uint32_t depth_u = __float_as_uint(depth);
    // For negative depths, flip all bits to preserve order
    if (depth_u >> 31)
        depth_u = ~depth_u;
    else
        depth_u ^= 0x80000000u;
    return ((uint64_t)tile_id << 32) | depth_u;
}

/**
 * @brief CUDA kernel to assign splats to screen tiles and emit
 * key-value pairs for sorting. 
 * 
 * One thread is launched per splat.
 * 
 * Each thread computes the screen-space bounding box of its splat
 * based on position and covariance, determines which tiles it overlaps,
 * and emits a key-value pair
 * 
 * @param[in] pos_x         X positions of splats in NDC [-1, 1]
 * @param[in] pos_y         Y positions of splats in NDC [-1, 1]
 * @param[in] pos_z         Z positions of splats in NDC [-1, 1]
 * @param[in] cov_a         Covariance matrix element a (cxx)
 * @param[in] cov_b         Covariance matrix element b (cxy)
 * @param[in] cov_d         Covariance matrix element d (cyy)
 * @param[in] splat_count   Total number of splats in the scene
 * @param[out] d_keys       Output buffer for emitted keys (tile_id + depth) [max_pairs]
 * @param[out] d_values     Output buffer for emitted values (splat_id) [max_pairs]
 * @param[out] d_pair_count Atomic counter for emitted pairs
 *                          Holds the total number of pairs emitted by all threads
 * @param[in] max_pairs     Capacity of the d_keys and d_values buffers
 * @param[in] num_tiles_x   Number of tiles in X direction on the screen
 * @param[in] num_tiles_y   Number of tiles in Y direction on the screen
 * @param[in] screen_width  Screen width in pixels
 * @param[in] screen_height Screen height in pixels
 */
__global__ void tileAssignKernel(
    const float *__restrict__ pos_x,
    const float *__restrict__ pos_y,
    const float *__restrict__ pos_z,
    const float *__restrict__ cov_a,
    const float *__restrict__ cov_b,
    const float *__restrict__ cov_d,
    int splat_count,
    uint64_t *d_keys,
    uint32_t *d_values,
    uint32_t *d_pair_count,
    int max_pairs,
    int num_tiles_x,
    int num_tiles_y,
    int screen_width,
    int screen_height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= splat_count)
        return;

    float x = pos_x[i];
    float y = pos_y[i];
    float z = pos_z[i];

    if (fabsf(x) > 1.0f || fabsf(y) > 1.0f || fabsf(z) > 1.0f)
        return;

    float cxx = cov_a[i];
    float cxy = cov_b[i];
    float cyy = cov_d[i];

    float det = cxx * cyy - cxy * cxy;
    if (det <= 0.0f)
        return;

    float trace = cxx + cyy;
    float temp = fmaxf(0.0f, trace * trace - 4.0f * det);
    float lambda1 = 0.5f * (trace + sqrtf(temp));
    float lambda2 = 0.5f * (trace - sqrtf(temp));

    float max_radius = 3.0f * sqrtf(fmaxf(lambda1, lambda2));

    // pixel size in NDC
    float pixel_ndc_x = 2.0f / (float)screen_width;
    float pixel_ndc_y = 2.0f / (float)screen_height;
    // discard splats smaller than half a pixel
    float pixel_ndc = fmaxf(pixel_ndc_x, pixel_ndc_y);
    if (max_radius * 4.0f < pixel_ndc)
        return;

    // bounding box in NDC
    float min_x = x - max_radius;
    float max_x = x + max_radius;
    float min_y = y - max_radius;
    float max_y = y + max_radius;

    // convert NDC bounding box to tile range
    auto ndcToTileX = [&](float v) -> int
    {
        return (int)floorf((v + 1.0f) * 0.5f * (float)num_tiles_x);
    };
    auto ndcToTileY = [&](float v) -> int
    {
        return (int)floorf((v + 1.0f) * 0.5f * (float)num_tiles_y);
    };

    int tx0 = max(ndcToTileX(min_x), 0);
    int tx1 = min(ndcToTileX(max_x), num_tiles_x - 1);
    int ty0 = max(ndcToTileY(min_y), 0);
    int ty1 = min(ndcToTileY(max_y), num_tiles_y - 1);

    // emit one pair per overlapping tile
    for (int ty = ty0; ty <= ty1; ty++)
    {
        for (int tx = tx0; tx <= tx1; tx++)
        {
            uint32_t tile_id = (uint32_t)(ty * num_tiles_x + tx);
            uint64_t key = makeKey(tile_id, z);

            // claim a slot
            uint32_t slot = atomicAdd(d_pair_count, 1u);
            if (slot >= (uint32_t)max_pairs)
            {
                // buffer full, back off (shouldn't happen with generous max_pairs)
                atomicSub(d_pair_count, 1u);
                return;
            }

            d_keys[slot] = key;
            d_values[slot] = (uint32_t)i;
        }
    }
}

void launchTileAssign(
    const GaussianParams &gaussians,
    uint64_t *d_keys,
    uint32_t *d_values,
    uint32_t *d_pair_count,
    int max_pairs,
    int num_tiles_x,
    int num_tiles_y,
    int width,
    int height)
{
    if (gaussians.count == 0)
        return;

    int threads = 256;
    int blocks = (gaussians.count + threads - 1) / threads;
    tileAssignKernel<<<blocks, threads>>>(
        gaussians.pos_x, gaussians.pos_y, gaussians.pos_z,
        gaussians.cov_a, gaussians.cov_b, gaussians.cov_d,
        gaussians.count,
        d_keys,
        d_values,
        d_pair_count,
        max_pairs,
        num_tiles_x,
        num_tiles_y,
        width,
        height
    );
    cudaDeviceSynchronize();
}
