#pragma once

// Configuration for MCMC-based Gaussian density control.
// Intentionally dependency-free so it can be included anywhere (app, CLI, fitter)
// without pulling in CUDA or Gaussian internals.
struct MCMCConfig {
    bool  enabled        = false;

    // Steps before density control starts.
    int   warmup_steps   = 500;

    // How often (in optimizer steps) to run a density control pass.
    int   interval       = 100;

    // L1 regularization strength added to grad_logit_opacity every step.
    // Gradient contribution = lambda * sigmoid(logit) * (1 - sigmoid(logit)).
    // This is a continuous pull toward zero opacity; weak Gaussians fade out
    // organically and become candidates for relocation.
    float opacity_reg_lambda = 1e-6f;

    // Gaussians whose opacity is below this are relocated each pass.
    float dead_threshold    = 0.001f;

    // Max fraction of all Gaussians that can be relocated in a single pass.
    // Prevents overcrowding near a few parents when many die at once.
    float max_reloc_fraction = 0.01f;

    // Std-dev of Gaussian positional noise added to relocated Gaussians.
    // Tune relative to your scene scale.
    float perturb_std    = 1.0f;

    // On relocation, both parent and child are set to parent_opacity * split_factor.
    // Prevents combined opacity from spiking above the original parent's level.
    float split_factor   = 0.5f;

    // Brownian noise added directly to parameters every step (Langevin dynamics).
    // Prevents Gaussians from locking into degenerate configurations. Set to 0 to disable.
    float pos_noise_std   = 0.05f;  // world-space units
    float scale_noise_std = 0.004f;  // log-scale
};
