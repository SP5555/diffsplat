# Archive

Superseded rasterizer implementations. Not wired into CMakeLists.txt; kept for
reference only.

Evolution:
- v1 used an `n_contrib` stack (silent MAX_CONTRIB
truncation)
- v2 switched to `last_ids` (no limit) → `rasterize_layer` cleaned that up 
- current `src/layers/gsplat_rasterize_layer` adds 2D blocks (one thread per pixel) and warp-level reduction before atomicAdd (~32x less atomic contention).
