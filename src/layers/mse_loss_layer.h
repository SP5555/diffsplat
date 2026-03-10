#pragma once
#include <cuda_runtime.h>

/**
 * @brief Calculates MSE loss between rendered image and target,
 * and its gradient w.r.t. pixel colors.
 */
class MSELossLayer
{
public:
    ~MSELossLayer() { free(); }

    void allocate(int width, int height);
    void free();
    void zero_grad();

    // wiring
    void setInput(const float *pixels)  { input = pixels; }
    void setTarget(const float *target) { d_target_pixels = target; }
    const float *getGradInput() const   { return d_grad_pixels; }

    float forward();
    void backward();

private:
    // not owned
    const float *input           = nullptr;
    const float *d_target_pixels = nullptr;

    // owned
    size_t numPixels = 0;
    float *d_grad_pixels   = nullptr;  // [H*W*3]
    float *d_loss          = nullptr;  // [1]
};
