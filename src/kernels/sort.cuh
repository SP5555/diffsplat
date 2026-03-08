#pragma once
#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Sorts key-value pairs in place using CUB radix sort.
 * 
 * @param[in] d_keys            Input keys on the device
 * @param[in] d_values          Input values on the device
 * @param[out] d_keys_sorted    Output keys on the device
 * @param[out] d_values_sorted  Output values on the device
 * @param[in] pair_count        Number of valid pairs to sort
 * @param[in,out] d_temp        Temporary storage on the device
 * @param temp_bytes 
 */
void launchSort(
    uint64_t *d_keys,
    uint32_t *d_values,
    uint64_t *d_keys_sorted,
    uint32_t *d_values_sorted,
    uint32_t pair_count,
    void **d_temp,     // pointer to temp buffer pointer (managed here)
    size_t *temp_bytes // current allocated temp size (updated if needed)
);

/**
 * @brief Builds tile ranges from sorted keys.
 * 
 * Required for the forward rasterizer to know
 * which slice of the sorted array belongs to each tile.
 * 
 * @param[in] d_keys_sorted     Input keys on the device, sorted by tile_id and depth
 * @param[out] d_tile_ranges    pair of uint32 per tile
 *                              {start, end} indices into sorted arrays.
 *                              int2.x = start, int2.y = end (exclusive)
 * @param[in] pair_count        Number of valid pairs in the sorted arrays
 * @param[in] num_tiles         Total number of tiles on the screen
 */
void launchBuildTileRanges(
    const uint64_t *d_keys_sorted,
    int2 *d_tile_ranges,
    uint32_t pair_count,
    int num_tiles
);
