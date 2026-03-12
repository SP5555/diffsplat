#pragma once
#include <cuda_runtime.h>

#include "layer.h"

/**
 * @brief Calculates MSE loss between rendered image and target,
 * and its gradient w.r.t. pixel colors.
 */
class MSELossLayer : public Layer
{
public:
    ~MSELossLayer() { free(); }

    void allocate(int width, int height);
    void forward()      override;
    void backward()     override;
    void zero_grad()    override;
    void free()         override;

    // wiring
    void setInput(const float *pixels)  { input = pixels; }
    void setTarget(const float *target) { d_target_pixels = target; }
    const float *getGradInput() const   { return d_grad_pixels; }

    float getLoss() const;

private:
    /* ---- forward input (not owned) ---- */
    const float *input           = nullptr;  // rendered pixels [H*W*3]
    const float *d_target_pixels = nullptr;  // target image    [H*W*3]
    
    /* ---- forward output (owned) ---- */
    float *d_loss = nullptr;  // scalar loss buffer [1]

    /* ---- backward input (not owned) ---- */
    // N/A

    /* ---- backward output (owned) ---- */
    float *d_grad_pixels = nullptr;  // dL/d_pixels [H*W*3]

    /* ---- config ---- */
    size_t num_pixels = 0;
};
