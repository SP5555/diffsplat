#pragma once
#include <cuda_runtime.h>
#include "../gaussian/gaussian.h"
#include <cstdint>

/**
 * @brief Assigns Gaussians to screen-space tiles.
 *
 * For each splat, compute the 3-sigma bounding box in NDC space and
 * emits one (key, value) pair for each tile the bounding box overlaps.
 *
 * The key encodes the tile index and depth so a subsequent radix sort
 * can group by tile and sort by depth for correct compositing order.
 *
 * - `key = [ tile_id (upper 32 bits) | depth_as_uint (lower 32 bits) ]`
 * 
 * - `value = splat_id (uint32)`
 *
 * @param[in] gaussiansSplat    Splat parameters on device
 * @param[out] d_keys           Key buffer for emitted pairs [max_pairs]
 * @param[out] d_values         Value buffer for emitted pairs [max_pairs]
 * @param[out] d_pair_count     Single uint32 on the device, atomically incremented
 *                              Read back after launch to get the true pair count
 *                              before passing to the radix sort.
 * @param[in] max_pairs         Capacity of the d_keys and d_values buffers
 *                              Pairs are dropped if the count exceeds this.
 * @param[in] num_tiles_x       Number of tiles in X direction on the screen
 * @param[in] num_tiles_y       Number of tiles in Y direction on the screen
 *                              Please size this generously!
 * @param[in] screen_width      Screen width in pixels
 * @param[in] screen_height     Screen height in pixels
 */
void launchTileAssign(
    const GaussianParams &gaussians,
    uint64_t *d_keys,
    uint32_t *d_values,
    uint32_t *d_pair_count,
    int max_pairs,
    int num_tiles_x,
    int num_tiles_y,
    int screen_width,
    int screen_height
);
