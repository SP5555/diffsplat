#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "../gaussian/gaussian.h"

/**
 * @brief Performs the backward pass, computing gradients for all Gaussian parameters.
 * 
 * - N = number of splats
 * 
 * - M = number of [Key, Value] pairs.
 * 
 * Key is [tile ID | depth] and Value is splat ID.
 * M is approximately N but can be larger due to same splat in multiple tiles.
 * 
 * @param[in] gaussians         Current Gaussian parameters
 * @param[out] opt_state        Output gradients for all Gaussian parameters
 * @param[in] d_target_pixels   Target pixel colors [H*W*3]
 * @param[in] d_values_sorted   Sorted contributing splat indices
 *                              for each tile [M]
 * @param[in] d_tile_ranges     Tile start/end indices
 *                              in d_values_sorted [num_tiles]
 * @param[in] d_pixels          Rendered pixel colors [H*W*3]
 * @param[in] d_T_final         Final transmittance values [H*W]
 * @param[in] d_n_contrib       Number of contributing splats
 *                              for each pixel [H*W]
 * @param[in] num_tiles_x       Number of tiles in x direction
 * @param[in] num_tiles_y       Number of tiles in y direction
 * @param[in] screen_width      Screen width in pixels
 * @param[in] screen_height     Screen height in pixels
 */
void launchBackward(
    const GaussianParams &gaussians,
    const GaussianOptState &opt_state,
    const float *d_target_pixels,
    const uint32_t *d_values_sorted,
    const int2 *d_tile_ranges,
    const float *d_pixels,
    const float *d_T_final,
    const int *d_n_contrib,
    int num_tiles_x,
    int num_tiles_y,
    int screen_width,
    int screen_height
);