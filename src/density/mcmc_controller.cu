#include "mcmc_controller.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

#include <cuda_runtime.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
#include "../cuda/math_ops.cuh"
#include "../utils/logs.h"

/* ===== ===== Kernels ===== ===== */

// L1 regularization: adds lambda * s*(1-s) to grad_logit_opacity each step.
// This is d/dx [lambda * sigmoid(x)] -- a continuous pull toward zero opacity.
// The gradient is largest at mid-opacity and small at extremes, so strong
// Gaussians are nudged gently while near-dead ones are pushed out faster.
__global__ void opacityRegKernel(
    float*       grad_logit_opacity,
    const float* logit_opacity,
    float        lambda,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sigmoid(logit_opacity[i]);
    grad_logit_opacity[i] += lambda * s * (1.f - s);
}

// Box-Muller via Wang hash -- no curand dependency.
__device__ float2 hashNormal(unsigned int seed, unsigned int idx)
{
    unsigned int u = (seed + 1u) ^ (idx * 2654435761u);
    u ^= u >> 16; u *= 0x45d9f3bu; u ^= u >> 16;
    unsigned int v = (seed + 2u) ^ (idx * 1664525u + 1013904223u);
    v ^= v >> 16; v *= 0x45d9f3bu; v ^= v >> 16;
    float f1 = (float)(u & 0xffffffu) * (1.f / (float)0x1000000u) + 1e-7f;
    float f2 = (float)(v & 0xffffffu) * (1.f / (float)0x1000000u);
    float r   = sqrtf(-2.f * logf(f1));
    float th  = 6.28318530f * f2;
    return {r * cosf(th), r * sinf(th)};
}

// Add iid Gaussian noise directly to position parameters (Brownian motion).
// Prevents dense clusters from permanently locking in place.
__global__ void positionNoiseKernel(
    float*       px, float* py, float* pz,
    float        std,
    unsigned int seed,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 na = hashNormal(seed,        (unsigned int)i);
    float2 nb = hashNormal(seed + 179u, (unsigned int)i);
    px[i] += na.x * std;
    py[i] += na.y * std;
    pz[i] += nb.x * std;
}

// Add iid Gaussian noise directly to log-scale parameters (Brownian motion).
// In log-space this is a multiplicative perturbation: exp(log_s + noise).
__global__ void scaleNoiseKernel(
    float*       sx, float* sy, float* sz,
    float        std,
    unsigned int seed,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 na = hashNormal(seed,        (unsigned int)i);
    float2 nb = hashNormal(seed + 251u, (unsigned int)i);
    sx[i] += na.x * std;
    sy[i] += na.y * std;
    sz[i] += nb.x * std;
}

// Scatter-write precomputed logit values to specific slots (used for parent splits).
__global__ void setLogitsKernel(
    float*       logit_opacity,
    const int*   indices,
    const float* new_logits,
    int          count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    logit_opacity[indices[i]] = new_logits[i];
}

// Copy full parent state into each dead slot, add small position perturbation.
// Child opacity is set to parent_opacity * split_factor (not the raw parent logit).
__global__ void relocateKernel(
    float*       px,  float* py,  float* pz,
    float*       sx,  float* sy,  float* sz,
    float*       rw,  float* rx,  float* ry, float* rz,
    float*       sh_dc_r, float* sh_dc_g, float* sh_dc_b,
    float*       logit_opacity,
    const int*   dead,
    const int*   parent,
    int          count,
    float        perturb_std,
    float        split_factor,
    unsigned int seed)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int d = dead[t];
    int p = parent[t];

    float2 na = hashNormal(seed,        (unsigned int)t);
    float2 nb = hashNormal(seed + 97u,  (unsigned int)t);

    px[d] = px[p] + na.x * perturb_std;
    py[d] = py[p] + na.y * perturb_std;
    pz[d] = pz[p] + nb.x * perturb_std;

    sx[d] = sx[p]; sy[d] = sy[p]; sz[d] = sz[p];
    rw[d] = rw[p]; rx[d] = rx[p]; ry[d] = ry[p]; rz[d] = rz[p];

    sh_dc_r[d] = sh_dc_r[p];
    sh_dc_g[d] = sh_dc_g[p];
    sh_dc_b[d] = sh_dc_b[p];

    float parent_op = sigmoid(logit_opacity[p]);
    float child_op  = fmaxf(parent_op * split_factor, 0.001f);
    logit_opacity[d] = logf(child_op / (1.f - child_op));
}

/* ===== ===== MCMCController ===== ===== */

void MCMCController::init(const MCMCConfig& cfg_, int n_gaussians,
                          const Gaussian3DGroupIndices& group_idx_)
{
    cfg      = cfg_;
    n        = n_gaussians;
    group_idx = group_idx_;

    h_opacity.resize(n);
    d_dead.allocate(n);
    d_parent.allocate(n);
    d_unique_parents.allocate(n);
    d_parent_new_logits.allocate(n);
}

void MCMCController::preOptimizerStep(
    const Gaussian3DParams& params,
    Gaussian3DGrads&        grads,
    int                     step)
{
    applyRegularization(params, grads);
}

void MCMCController::postOptimizerStep(
    int               step,
    Gaussian3DParams& params,
    Adam&             adam)
{
    applyParamNoise(params);
    maybeStep(step, params, adam);
}

void MCMCController::applyRegularization(const Gaussian3DParams& params, Gaussian3DGrads& grads)
{
    if (!cfg.enabled) return;
    if (cfg.opacity_reg_lambda <= 0.f) return;
    int blocks = divRoundUp(n, BLOCK_SIZE);
    opacityRegKernel<<<blocks, BLOCK_SIZE>>>(
        grads.grad_logit_opacity, params.logit_opacity, cfg.opacity_reg_lambda, n);
    CUDA_SYNC_CHECK();
}

void MCMCController::applyParamNoise(Gaussian3DParams& params)
{
    if (!cfg.enabled) return;
    int blocks = divRoundUp(n, BLOCK_SIZE);
    if (cfg.pos_noise_std > 0.f) {
        positionNoiseKernel<<<blocks, BLOCK_SIZE>>>(
            params.pos_x, params.pos_y, params.pos_z,
            cfg.pos_noise_std, rng_seed++, n);
    }
    if (cfg.scale_noise_std > 0.f) {
        scaleNoiseKernel<<<blocks, BLOCK_SIZE>>>(
            params.scale_x, params.scale_y, params.scale_z,
            cfg.scale_noise_std, rng_seed++, n);
    }
    CUDA_SYNC_CHECK();
}

void MCMCController::maybeStep(int step, Gaussian3DParams& params, Adam& adam)
{
    if (!cfg.enabled) return;
    if (step <= cfg.warmup_steps) return;
    if (cfg.interval <= 0) return;
    if (step % cfg.interval != 0) return;
    run(step, params, adam);
}

void MCMCController::run(int step, Gaussian3DParams& params, Adam& adam)
{
    // 1. Download current opacities and find dead Gaussians.
    CUDA_CHECK(cudaMemcpy(h_opacity.data(), params.logit_opacity,
                          n * sizeof(float), cudaMemcpyDeviceToHost));

    h_dead.clear();
    float total_op = 0.f;
    for (int i = 0; i < n; i++) {
        float s = 1.f / (1.f + std::exp(-h_opacity[i]));
        h_opacity[i] = s;
        total_op += s;
        if (s < cfg.dead_threshold)
            h_dead.push_back(i);
    }

    int dead_count = (int)h_dead.size();
    if (dead_count == 0) return;

    // 3. Cap relocations to avoid overcrowding.
    int total_dead = dead_count; // save pre-cap count for logging
    int max_reloc  = std::max(1, (int)(cfg.max_reloc_fraction * (float)n));
    if (dead_count > max_reloc) {
        std::mt19937 cap_rng(rng_seed++);
        std::shuffle(h_dead.begin(), h_dead.end(), cap_rng);
        h_dead.resize(max_reloc);
        dead_count = max_reloc;
    }

    // 4. Build CDF over post-knockdown opacities and sample a parent for each dead.
    std::vector<float> cdf(n);
    float cumsum = 0.f;
    for (int i = 0; i < n; i++) {
        cumsum += h_opacity[i] / total_op;
        cdf[i]  = cumsum;
    }
    cdf[n - 1] = 1.f;

    h_parent.resize(dead_count);
    std::mt19937 sample_rng(rng_seed++);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int j = 0; j < dead_count; j++) {
        float u = dist(sample_rng);
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] < u) lo = mid + 1; else hi = mid;
        }
        h_parent[j] = lo;
    }

    // 5. Upload index pairs and scatter-write new state on GPU.
    CUDA_CHECK(cudaMemcpy(d_dead,   h_dead.data(),   dead_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent, h_parent.data(), dead_count * sizeof(int), cudaMemcpyHostToDevice));

    // Build unique-parent list on CPU: each parent gets opacity * split_factor,
    // computed once regardless of how many children it spawns.
    std::unordered_map<int, float> parent_map;
    for (int j = 0; j < dead_count; j++) {
        int p = h_parent[j];
        if (parent_map.find(p) == parent_map.end()) {
            float s     = h_opacity[p];
            float new_s = fmaxf(s * cfg.split_factor, 0.001f);
            parent_map[p] = logf(new_s / (1.f - new_s));
        }
    }
    h_unique_parents.clear();
    h_parent_new_logits.clear();
    for (auto& kv : parent_map) {
        h_unique_parents.push_back(kv.first);
        h_parent_new_logits.push_back(kv.second);
    }
    int unique_count = (int)h_unique_parents.size();

    // Relocate children (each child starts at parent_opacity * split_factor).
    int rblocks = divRoundUp(dead_count, BLOCK_SIZE);
    relocateKernel<<<rblocks, BLOCK_SIZE>>>(
        params.pos_x,   params.pos_y,   params.pos_z,
        params.scale_x, params.scale_y, params.scale_z,
        params.rot_w,   params.rot_x,   params.rot_y, params.rot_z,
        params.sh_dc_r, params.sh_dc_g, params.sh_dc_b,
        params.logit_opacity,
        d_dead, d_parent, dead_count,
        cfg.perturb_std, cfg.split_factor, rng_seed++);

    // Knock parent opacities down in a separate pass (race-free: each parent
    // appears exactly once in h_unique_parents).
    CUDA_CHECK(cudaMemcpy(d_unique_parents,    h_unique_parents.data(),    unique_count * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent_new_logits, h_parent_new_logits.data(), unique_count * sizeof(float), cudaMemcpyHostToDevice));
    int pblocks = divRoundUp(unique_count, BLOCK_SIZE);
    setLogitsKernel<<<pblocks, BLOCK_SIZE>>>(
        params.logit_opacity, d_unique_parents, d_parent_new_logits, unique_count);
    CUDA_SYNC_CHECK();

    // 6. Zero Adam moments for relocated slots.
    adam.zeroSlotMoments(d_dead, dead_count, n);
    if (group_idx.sh_rest_r >= 0) {
        adam.zeroGroupMoments(group_idx.sh_rest_r);
        adam.zeroGroupMoments(group_idx.sh_rest_g);
        adam.zeroGroupMoments(group_idx.sh_rest_b);
    }

    char buf[128];
    if (total_dead > dead_count)
        snprintf(buf, sizeof(buf),
            "step %d: relocated %d, split %d parents  [cap hit, %d total dead]",
            step, dead_count, unique_count, total_dead);
    else
        snprintf(buf, sizeof(buf),
            "step %d: relocated %d, split %d parents",
            step, dead_count, unique_count);
    log_info("MCMC", buf);
}
