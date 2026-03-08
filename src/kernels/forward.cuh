#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "../gaussian/gaussian.h"

/**
 * @brief Forward rasterize gaussians to screen pixels.
 * 
 * Iterates over sorted splats per tile, alpha compositing front-to-back.
 * Stores final pixel colors and T values for backward pass.
 * 
 * @param gaussians         Gaussian parameters on the device
 * @param d_values_sorted   Sorted splat IDs on the device, sorted by tile_id and depth
 * @param d_tile_ranges     Tile ranges for each tile
 * @param d_pixels          Output pixel colors (RGB)
 * @param d_T_final         Final T values for backward pass
 * @param d_n_contrib       Number of contributing splats per pixel
 * @param num_tiles_x       Number of tiles in x direction
 * @param num_tiles_y       Number of tiles in y direction
 * @param screen_width      Screen width in pixels
 * @param screen_height     Screen height in pixels
 */
void launchForward(
    const GaussianParams &gaussians,
    const uint32_t *d_values_sorted,
    const int2 *d_tile_ranges,
    float *d_pixels,
    float *d_T_final,
    int *d_n_contrib,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height
);
