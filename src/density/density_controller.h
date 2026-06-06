#pragma once

#include "../optimizers/adam.h"
#include "../types/gaussian3d.h"

// Abstract interface for Gaussian density control.
//
// ImageFitter calls exactly two methods per step:
//   preOptimizerStep  -- between backward() and optimizer.step()
//                        use for gradient modifications (reg terms, etc.)
//   postOptimizerStep -- after optimizer.step()
//                        use for parameter modifications and periodic passes
//
// Both have default no-op implementations so a concrete algorithm only
// overrides what it needs. To add a new algorithm, inherit this class,
// implement the relevant methods, and instantiate it in ImageFitter::initLayers.
class DensityController
{
public:
    virtual ~DensityController() = default;

    virtual void preOptimizerStep(
        const Gaussian3DParams& params,
        Gaussian3DGrads&        grads,
        int                     step) {}

    virtual void postOptimizerStep(
        int               step,
        Gaussian3DParams& params,
        Adam&             adam) {}
};
