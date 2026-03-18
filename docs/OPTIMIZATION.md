# Optimization Notes

This document records the profiling-driven optimization process for the tile-based Gaussian splat rasterizer, including what was tried, what worked, what didn't, and why.

---

## Methodology

All profiling was done with NVIDIA Nsight Systems (`nsys`) on a training loop running forward + backward + Adam optimization per step. FPS numbers are eyeballed (yeah) from a live counter over a sustained run. Kernel timing numbers are from `nsys` averaged over ~500 iterations.

**Test scene:** uniformly initialized Gaussian splats in world/NDC space, camera transform is near-identity. This is the "worst" case for splat clustering (or maybe the "best"?), every tile receives roughly equal splat density. Real trained scenes with clustered splats show larger gains (noted where relevant).

---

## Profile Before Touching Anything

Before optimizing, the full kernel breakdown at 80k splats looked like this:

| Time % | Kernel |
|--------|--------|
| 56.3% | `backwardKernel` (from Rasterizer) |
| 22.5% | `rasterizeKernel` |
| 17.5% | `mseLossKernel` |
| 2.0%  | `mseGradKernel` |
| 0.5%  | CUB radix sort |
| 0.5%  | `adamKernel` |
| 0.2%  | `covForwardKernel` |
| 0.1%  | `covBackwardKernel` |

Key takeaways:
- `covForwardKernel` and `covBackwardKernel` were the original optimization target (quaternion to rotation matrix math). They account for **0.3% of runtime combined**. So, **not worth touching**.
- `mseLossKernel` at 17.5% was low-hanging fruit.
- `rasterizeKernel` and `backwardKernel` at 56.3% and 22.5% each, were the two main culprits.

---

## MSE Loss Kernel Optimization (Free 7-12% Gain)

**Problem:** `mseLossKernel` used a single `atomicAdd` to a scalar from every pixel thread simultaneously. At 1080p that's ~2M threads all serialized and fighting over one memory address (the definition of a bottleneck can't get any more accurate than this).

**Fix:** Two-stage shared memory reduction. Each thread accumulates its pixel's contribution locally, then threads within a block reduce into shared memory together, and only one thread per block issues a single `atomicAdd` to global memory. This reduces global atomic operations from one per pixel down to one per block (one per 256 pixels).

```C++
__shared__ float sdata[BLOCK_SIZE];
// ... each thread fills sdata[tid] with its pixel loss ...

// reduce within block using shared memory
for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
        sdata[tid] += sdata[tid + stride];
    __syncthreads();
}

// one atomicAdd per block instead of per thread
if (tid == 0)
    atomicAdd(loss, sdata[0]);
```

**Result:** `mseLossKernel` dropped from 17.5% to ~2% of kernel runtime. FPS improvement scaled with resolution (7% at 720p, ~12% at 2000×1200) because the original atomic contention scaled directly with pixel count.

---

## Rasterizer Architecture (Naive Baseline)

The original rasterizer launched one thread per pixel with a 2D `dim3(16,16)` block layout. Each thread independently walked its tile's sorted splat list, loading splat data via scattered global reads (`ndc_x[sid]`, `ndc_cxx[sid]`, etc. where `sid` is sorted by depth, not by splat ID).

**Forward:**
- Scattered global reads for every splat, every thread
- No data reuse between threads in the same tile

**Backward:**
- Two passes per pixel: forward sweep to collect IDs of contributed splats to the corresponding pixel into a 256-element per-thread array, then reverse walk using T_final division trick
- After walking though all the splats in reverse order, each thread performs direct `atomicAdd` to global gradient buffers **for every contributing splat of every pixel**

---

## Tile-Cooperative Rasterizer

**Key insight:** all pixels within a tile share the same splat list. If one
block owns one tile, all threads can collaborate to load splat data into shared
memory once per chunk, and every thread reads from there instead of issuing
independent scattered global reads.

**Architecture change:** one CUDA block per tile (1D, 256 threads per block)
instead of one thread per pixel (2D, 16x16 threads per block). Since tiles can contain more pixels
than `BLOCK_THREADS`, pixels are processed in sequential batches within each block.

Three variants of the cooperative kernel were benchmarked against the original:

### Variant A: Naive (original)

- Forward
  - Each thread independently walks its tile's sorted splat list
  - Splat data (`ndc_x[sid]`, `ndc_cxx[sid]`, etc.) is fetched via scattered global reads, `sid` is sorted by depth, not by splat ID, so accesses are non-coalesced and repeated identically across all threads in the tile
- Backward
  - Two-pass per pixel: forward sweep collects contributing splat IDs into a per-thread `contrib_indices[256]` array, then reverse walk computes gradients
  - `contrib_indices` array easily exceeds register capacity ([Official Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)) and spills to local memory (which is physically global memory), adding hidden read/write overhead
  - Every contributing splat of every pixel issues direct `atomicAdd` to global gradient buffers, at high splat counts this causes heavy traffic jam since many pixels in the same tile overlap the same splats

### Variant B: Full Cooperative

- Forward
  - Splats streamed through shared memory in chunks of `CHUNK_SIZE`
  - All threads collaboratively load each chunk (coalesced reads), then each thread composites its pixel against the cached chunk, eliminating repeated scattered global reads
- Backward
  - Same collaborative loading, chunks streamed in reverse order
  - No `contrib_indices`, each thread must test every splat in every chunk to determine contribution, recomputing full geometry each time. A pixel that composites only 5 splats still pays the geometry test cost for every splat in the tile
  - Gradients accumulate into shared memory atomics (`sh_grad_*`) per chunk, then flush to global with one `atomicAdd` per splat per chunk, reducing global atomic pressure proportional to tile pixel count
  - **Critical downside:** pixel batching replays the entire splat stream once per batch. At high splat density with uniform distribution, this replay cost dominates, making Variant B slower than naive despite the shared memory improvements

### Variant C: Hybrid (current, default)

- Forward (identical to Variant B)
- Backward
  - Two-pass structure like Variant A, but both passes read from shared memory rather than global, no scattered global reads at any point
  - Forward sweep: collaboratively load splats chunk by chunk into shared memory, each thread scans the chunk for contributing splats and records their absolute position in `values_sorted` into `contrib_pos[]`. Geometry tests run once here, against shared memory
  - Reverse sweep: chunks loaded in reverse, each thread walks `contrib_pos[]` backward using a cursor. Non-contributing splats are skipped with a single boundary check, no geometry recomputation. Splat data indexed directly from shared memory.
  - Gradients accumulate into `sh_grad_*` and flush to global once per chunk, same as Variant B
  - `contrib_cursor` naturally terminates once all contributors are processed, pixels that go opaque early (T < T_THRESHOLD) pay nothing for remaining chunks beyond the mandatory collaborative load and sync

---

## Benchmark Results

### FPS (forward + backward + Adam, uniform splat distribution, tested with `imgfitapp`)

| Splat Count | Naive | Full Cooperative | Hybrid  | Hybrid vs Naive |
|-------------|-------|------------------|---------|-----------------|
| 10k         | 90-95 | 79-82            | 90-92   | ~0%             |
| 80k         | 39-40 | 30-31            | 44-45   | +12.5%          |
| 200k        | 18.0  | 13.9             | 21.5    | +19.4%          |
| 800k        | 8.2   | 6.8              | 8.2-8.5 | +2-4%           |

### Kernel timing at 400k splats (uniform distribution)

| Kernel    | Naive  | Hybrid | Difference |
|-----------|--------|--------|-------|
| `backwardKernel` (from Rasterizer)  | 67.5ms | 69.8ms | hybrid +3.4% slower |
| `rasterizeKernel`   | 27.9ms | 19.4ms | hybrid -30.4% faster |
| **total** | **95.4ms** | **89.2ms** | **hybrid -6.5% faster** |

### Real clustered scene (forward-only, variable splat density, tested with `plyviewapp`)

| Kernel            | Naive  | Hybrid | Delta |
|-------------------|--------|--------|-------|
| `rasterizeKernel` | 15.2ms | 8.8ms  | **-42% faster** |
| `tileAssign`      | 2.84ms | 2.77ms | ~same |
| CUB sort          | 0.52ms | 0.52ms | ~same |

---
