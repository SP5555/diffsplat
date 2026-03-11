#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "splat2d_params.h"

/**
 * @brief Rasterizes 2D Gaussian splats into a pixel buffer,
 * using a tiled forward rendering approach.
 * 
 * Forward pass: Each splat is assigned to all tiles it overlaps.
 * The resulting (tile_id | depth, splat_id) pairs are radix sorted
 * so splats are processed front-to-back per tile. Each pixel then
 * alpha-composites its tile's splats in depth order. Final transmittance
 * `T_final` and contributing splat count n_contrib are saved for the
 * backward pass.
 * 
 * Backward pass: Since the sort is not differentiable, gradients do
 * not flow through the sorting step. Instead, each pixel re-computes
 * its forward compositing in reverse using the `T_final` division trick
 * (`T_before` is recovered by dividing `T_after` by `(1-alpha)`), avoiding
 * the need to store per-splat intermediate transmittance values.
 * Gradients w.r.t. splat parameters are computed on the fly and
 * accumulated via `atomicAdd` since multiple pixels may contribute
 * back to the same splat.
 */
class RasterizeLayer
{
public:
    ~RasterizeLayer() { free(); }

    void allocate(int width, int height, int num_tiles_x, int num_tiles_y,
                  int max_pairs, int count);
    void free();
    void zero_grad();

    // wiring
    void setInput(const Splat2DParams *params) { input = params; }
    void setGradOutput(const float *grad)      { gradOutput = grad; }
    const float        *getOutput() const      { return d_pixels; }
    const Splat2DGrads &getGradInput() const   { return gradInput; }

    // tile assign + sort + rasterize -> pixels
    void forward();
    void backward();

    // debug
    const float *getGradOutput() const { return gradOutput; }

private:
    // not owned
    const Splat2DParams *input      = nullptr;
    const float         *gradOutput = nullptr;

    // owned forward output
    float *d_pixels    = nullptr;  // [H*W*3]

    float *d_T_final   = nullptr;  // [H*W]
    int   *d_n_contrib = nullptr;  // [H*W]

    uint64_t *d_keys           = nullptr;
    uint32_t *d_values         = nullptr;
    uint64_t *d_keys_sorted    = nullptr;
    uint32_t *d_values_sorted  = nullptr;
    uint32_t *d_pair_count     = nullptr;
    int2     *d_tile_ranges    = nullptr;
    void     *d_sort_temp      = nullptr;
    size_t    sort_temp_bytes  = 0;

    Splat2DGrads gradInput;

    // config
    int screen_width = 0;
    int screen_height = 0;
    int numPixels   = 0;
    int num_tiles_x = 0;
    int num_tiles_y = 0;
    int max_pairs   = 0;
};
