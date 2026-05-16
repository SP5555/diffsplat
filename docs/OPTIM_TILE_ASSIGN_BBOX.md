# Tile Assignment Bounding Box Optimization

**Commit:** `2f0f930`

**File:** `src/layers/tile_rasterize_layer.cu` -- `tileAssignKernel`

**Guarantee:** strictly faster or equal, never slower.

---

## What Changed

The tile assignment kernel computes a 2D screen-space bounding box for each Gaussian to determine which tiles it overlaps. The old approach derived a circumscribed circle from the eigenvalues of the 2D covariance matrix; the new approach uses per-axis 3-sigma extents directly.

**Old -- circumscribed circle:**

```cpp
float trace   = cxx + cyy;
float temp    = fmaxf(0.f, trace * trace - 4.f * det);
float lambda2 = 0.5f * (trace + sqrtf(temp));

float pixel_ndc = fmaxf(2.f / screen_width, 2.f / screen_height);
if (3.f * sqrtf(lambda2) * 2.f >= pixel_ndc)
{
    float lambda1    = 0.5f * (trace - sqrtf(temp));
    float max_radius = 3.f * sqrtf(fmaxf(lambda1, lambda2));

    float min_x = x - max_radius, max_x = x + max_radius;
    float min_y = y - max_radius, max_y = y + max_radius;
```

**New -- tight axis-aligned box:**

```cpp
float extent_x = 3.f * sqrtf(cxx);
float extent_y = 3.f * sqrtf(cyy);

float pixel_ndc = fmaxf(2.f / screen_width, 2.f / screen_height);
if (fmaxf(extent_x, extent_y) * 2.f >= pixel_ndc)
{
    float min_x = x - extent_x, max_x = x + extent_x;
    float min_y = y - extent_y, max_y = y + extent_y;
```

## How It Was Found

The original circumscribed-circle approach was borrowed from a reference implementation. At some point the question came up: a scaling matrix that scales only along one axis only affects a single column of the covariance matrix. So it should be entirely possible to read the axis-aligned extents of the ellipse directly from the diagonal entries of the covariance matrix, without ever touching the principal axes. Turns out that is exactly right -- and the math in the next section confirms why.

## Why It's Correct

The 3-sigma ellipse in screen space is the set of points satisfying the Mahalanobis distance equal to 9. Let $\mathbf{d} = (dx,\, dy)^\top$ where $dx = x - \mu_x$, $dy = y - \mu_y$, and let $\Sigma$ be the 2D covariance:

```math
\Sigma = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{bmatrix}, \qquad \mathbf{d}^\top \Sigma^{-1} \mathbf{d} = 9
```

Expanding with the explicit 2x2 inverse ($\det\Sigma = \sigma_{xx}\sigma_{yy} - \sigma_{xy}^2$):

```math
\frac{\sigma_{yy}\, dx^2 - 2\sigma_{xy}\, dx\, dy + \sigma_{xx}\, dy^2}{\det\Sigma} = 9
```

This is exactly `dist2` evaluated in `compositeFwdKernel`. To find the x half-extent, fix $dx$ and ask: for what range of $dx$ does a real $dy$ exist on the ellipse? Rearranging as a quadratic in $dy$:

```math
\sigma_{xx}\, dy^2 - 2\sigma_{xy}\, dx\, dy + (\sigma_{yy}\, dx^2 - 9\det\Sigma) = 0
```

A real solution exists when the discriminant is non-negative:

```math
4\sigma_{xy}^2\, dx^2 - 4\sigma_{xx}(\sigma_{yy}\, dx^2 - 9\det\Sigma) \geq 0
```

```math
4(\sigma_{xy}^2 - \sigma_{xx}\sigma_{yy})\, dx^2 + 36\,\sigma_{xx}\det\Sigma \geq 0
```

Since $\sigma_{xy}^2 - \sigma_{xx}\sigma_{yy} = -\det\Sigma$:

```math
-4\det\Sigma\, dx^2 + 36\,\sigma_{xx}\det\Sigma \geq 0 \implies dx^2 \leq 9\,\sigma_{xx}
```

So $\sigma_{xy}$ cancels entirely, and the tight AABB half-extents are:

```math
|dx| \leq 3\sqrt{\sigma_{xx}}, \qquad |dy| \leq 3\sqrt{\sigma_{yy}}
```

> Visual: [Desmos -- ellipse, tight AABB, and circumscribed circle side by side](https://www.desmos.com/calculator/aakujicnkg)

The old approach used the major eigenvalue $\lambda_{\max}$ as a uniform radius, producing a circle that circumscribes the ellipse. For elongated Gaussians this significantly over-covers the bounding box, touching far more tiles than the Gaussian actually overlaps.

## Why It's Fast

Fewer tiles touched per Gaussian means:

- Fewer entries emitted into `values_sorted` (the depth-keyed tile list fed to CUB radix sort)
- Shorter per-tile splat lists, so `rasterizeKernel` and `backwardKernel` iterate less
- The savings compound at high splat counts and in real clustered scenes where many Gaussians are elongated

No extra compute -- the old path did two `sqrtf` calls and extra arithmetic for eigenvalues; the new path does two `sqrtf` calls total, one per axis.

## Results

FPS measured by eye from the live counter in `viewer`. No formal profiling (you know what, that's okay!).

| Scene                   | Platform | Before    | After     | Gain |
|-------------------------|----------|-----------|-----------|------|
| Cactus (~1.5M splats)   | Linux    | 30-32 FPS | 40-42 FPS | +33% |
| Playroom                | Linux    | 15-17 FPS | 30-32 FPS | +90% |
| Cactus (~464k splats)   | Windows  | ~90 FPS   | ~175 FPS  | +95% |

The playroom and 464k cactus gains are larger, likely because those scenes have more anisotropic (elongated) Gaussians where the circumscribed-circle overestimate is worst. Note: Windows numbers are not directly comparable to Linux -- Windows consistently shows higher FPS on the same hardware (honestly, I don't know why).
