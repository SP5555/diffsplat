#pragma once
#include <vector>

#include "../cuda/cuda_buffer.h"
#include "../density/density_controller.h"
#include "../density/mcmc_config.h"
#include "../optimizers/gaussian3d_adam.h"

// MCMC density controller. Implements DensityController with:
//   preOptimizerStep  -- L1 opacity regularization (continuous pull toward zero)
//   postOptimizerStep -- Brownian position noise + periodic relocation pass
class MCMCController : public DensityController
{
public:
    void init(const MCMCConfig& cfg, int n_gaussians,
              const Gaussian3DGroupIndices& group_idx);

    void preOptimizerStep(
        const Gaussian3DParams& params,
        Gaussian3DGrads&        grads,
        int                     step) override;

    void postOptimizerStep(
        int               step,
        Gaussian3DParams& params,
        Adam&             adam) override;

private:
    MCMCConfig             cfg;
    int                    n        = 0;
    Gaussian3DGroupIndices group_idx;
    unsigned int           rng_seed = 42u;

    std::vector<float> h_opacity;
    std::vector<int>   h_dead;
    std::vector<int>   h_parent;
    std::vector<int>   h_unique_parents;
    std::vector<float> h_parent_new_logits;
    CudaBuffer<int>    d_dead;
    CudaBuffer<int>    d_parent;
    CudaBuffer<int>    d_unique_parents;
    CudaBuffer<float>  d_parent_new_logits;

    void applyRegularization(const Gaussian3DParams& params, Gaussian3DGrads& grads);
    void applyParamNoise(Gaussian3DParams& params);
    void maybeStep(int step, Gaussian3DParams& params, Adam& adam);
    void run(int step, Gaussian3DParams& params, Adam& adam);
};
