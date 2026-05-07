#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "layer.h"
#include "../cuda/cuda_buffer.h"
#include "../types/splat2d.h"

/**
 * Tile-based differentiable Gaussian rasterizer.
 *
 * Forward: assigns splats to 16x16 pixel tiles, depth-sorts per tile with CUB
 * radix sort, then streams sorted splats in 256-element shared-memory batches for
 * front-to-back alpha compositing. Each pixel exits early when transmittance drops
 * below T_THRES. last_ids[pixel] records the sorted-list index of the final
 * contributing Gaussian for the backward pass.
 *
 * Backward: walks last_ids in reverse without replaying the full splat stream.
 * Warp-level shuffle reduction before global atomicAdd reduces atomic contention
 * by ~32x. The last_ids strategy removes the MAX_CONTRIB limit of an earlier
 * stack-based approach and avoids storing a per-pixel contributor list.
 */
class TileRasterizeLayer
    : public TypedLayer<Splat2DParams, CudaBuffer<float>, Splat2DGrads, CudaBuffer<float>>
{
public:
    ~TileRasterizeLayer() {}

    using TypedLayer::allocate;
    void allocate(int width, int height, int count);
    void forward()  override;
    void backward() override;

    void resize(int new_width, int new_height);
    uint32_t getVisibleCount();

private:
    /* ---- forward state saved for backward ---- */
    CudaBuffer<float>   d_render_alphas; // accumulated alpha per pixel [H*W]
    CudaBuffer<int32_t> d_last_ids;      // sorted-list index of last contributing Gaussian [H*W]

    /* ---- tile sorting internals (owned) ---- */
    CudaBuffer<uint64_t> d_isect_ids;
    CudaBuffer<uint32_t> d_gauss_ids;
    CudaBuffer<uint64_t> d_isect_ids_sorted;
    CudaBuffer<uint32_t> d_flatten_ids;
    CudaBuffer<uint32_t> d_n_isects;
    CudaBuffer<uint32_t> d_visible_count;
    CudaBuffer<int2>     d_tile_offsets;
    CudaBuffer<uint8_t>  d_sort_temp;
    size_t sort_temp_bytes = 0;

    /* ---- config ---- */
    int screen_width  = 0;
    int screen_height = 0;
    int num_pixels    = 0;
    int num_tiles_x   = 0;
    int num_tiles_y   = 0;
    int max_isects    = 0;
};
