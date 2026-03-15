#pragma once
#include <cuda_runtime.h>

#include "layer.h"
#include "../utils/cuda_utils.h"

/**
 * @brief Calculates MSE loss between rendered image and target,
 * and its gradient w.r.t. pixel colors.
 */
class MSELossLayer : public Layer
{
public:
    ~MSELossLayer() {}

    void allocate(int width, int height);
    void forward()      override;
    void backward()     override;
    void zero_grad()    override;

    // wiring
    void setInput(const float *pixels)  { input = pixels; }
    void setTarget(const float *target) { d_target_pixels = target; }
    float *getGradInput()               { return d_grad_pixels; }

    float getLoss() const;

private:
    /* ---- forward input (not owned) ---- */
    const float *input           = nullptr;  // rendered pixels [H*W*3]
    const float *d_target_pixels = nullptr;  // target image    [H*W*3]
    
    /* ---- forward output (owned) ---- */
    CudaBuffer<float> d_loss;  // scalar loss buffer [1]

    /* ---- backward input (not owned) ---- */
    // N/A

    /* ---- backward output (owned) ---- */
    CudaBuffer<float> d_grad_pixels;  // dL/d_pixels [H*W*3]

    /* ---- config ---- */
    size_t num_pixels = 0;
};
