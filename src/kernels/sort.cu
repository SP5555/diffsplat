#include "sort.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstring>
#include <stdio.h>

void launchSort(
    uint64_t *d_keys,
    uint32_t *d_values,
    uint64_t *d_keys_sorted,
    uint32_t *d_values_sorted,
    uint32_t pair_count,
    void **d_temp,
    size_t *temp_bytes)
{
    if (pair_count == 0)
        return;

    size_t required = 0;
    // query required temp storage size
    // this call does no sorting
    cub::DeviceRadixSort::SortPairs(
        nullptr, required,
        d_keys, d_keys_sorted,
        d_values, d_values_sorted,
        (int)pair_count);

    // reallocate temp buffer if needed
    if (required > *temp_bytes)
    {
        if (*d_temp)
            cudaFree(*d_temp);
        cudaMalloc(d_temp, required);
        *temp_bytes = required;
    }

    // actual sort
    cub::DeviceRadixSort::SortPairs(
        *d_temp, required,
        d_keys, d_keys_sorted,
        d_values, d_values_sorted,
        (int)pair_count);

    cudaDeviceSynchronize();
}

/**
 * @brief Builds tile ranges from sorted keys.
 * 
 * One thread is launched per key-value pair.
 * 
 * @param[in] d_keys_sorted     Input keys on the device, sorted by tile_id and depth
 * @param[out] d_tile_ranges    pair of uint32 per tile
 *                              {start, end} indices into sorted arrays.
 *                              int2.x = start, int2.y = end (exclusive)
 * @param[in] pair_count        Number of valid pairs in the sorted arrays
 * @param[in] num_tiles         Total number of tiles on the screen
 */
__global__ void buildTileRangesKernel(
    const uint64_t *keys_sorted,
    int2 *tile_ranges,
    uint32_t pair_count,
    int num_tiles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)pair_count)
        return;

    int tile_id = (int)(keys_sorted[i] >> 32);
    // invalid tile_id (should not happen, but just in case)
    if (tile_id < 0 || tile_id >= num_tiles)
        return;

    // first pair in this tile's run
    if (i == 0 || (int)(keys_sorted[i - 1] >> 32) != tile_id)
    {
        tile_ranges[tile_id].x = i; // start
    }

    // last pair in this tile's run
    if (i == (int)pair_count - 1 || (int)(keys_sorted[i + 1] >> 32) != tile_id)
    {
        tile_ranges[tile_id].y = i + 1; // end (exclusive)
    }
}

void launchBuildTileRanges(
    const uint64_t *d_keys_sorted,
    int2 *d_tile_ranges,
    uint32_t pair_count,
    int num_tiles)
{
    if (pair_count == 0)
        return;

    // Clear ranges
    // required to reset tiles with no splats to (0,0)
    cudaMemset(d_tile_ranges, 0, num_tiles * sizeof(int2));

    int threads = 256;
    int blocks = ((int)pair_count + threads - 1) / threads;
    buildTileRangesKernel<<<blocks, threads>>>(
        d_keys_sorted, d_tile_ranges, pair_count, num_tiles);
    cudaDeviceSynchronize();
}
