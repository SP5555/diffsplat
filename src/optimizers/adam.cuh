#pragma once
#include "../types/gaussian3d.h"

#define EPSILON 1e-8f

struct AdamConfig
{
    float lr_master     = 1e-3f;

    // different parameters may require different learning rates
    // because of different scalings. manually tune them!
    float lr_pos        = lr_master * 20.f;
    float lr_scale      = lr_master * 0.2f;
    float lr_rot        = lr_master * 0.1f;
    float lr_color      = lr_master * 4.f;
    float lr_opacity    = lr_master * 0.8f;

    float beta1         = 0.9f;
    float beta2         = 0.999f;
    float epsilon       = EPSILON;
};

/**
 * @brief Performs the Adam optimization step.
 * 
 * @param[in] gaussians         Current Gaussian parameters
 * @param[out] opt_state        Output gradients for all Gaussian parameters
 * @param[in] config            Adam optimization configuration
 * @param[in] step              Current optimization step number (starting from 1)
 */
void launchAdam(
    Gaussian3DParams &gaussians,
    const Gaussian3DOptState &opt_state,
    const AdamConfig &config,
    int step
);