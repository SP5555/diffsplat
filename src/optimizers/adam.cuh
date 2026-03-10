#pragma once
#include "../gaussian/gaussian.h"

#define EPSILON 1e-8f

struct AdamConfig
{
    float lr_master     = 1e-3f;

    // different parameters may require different learning rates
    // because of different scalings. manually tune them!
    float lr_pos        = lr_master * 20.f;
    float lr_cov        = lr_master * 20.f;
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
    GaussianParams &gaussians,
    GaussianOptState &opt_state,
    const AdamConfig &config,
    int step
);