#pragma once
#include "../gaussian/gaussian.h"
#include "splat2d_params.h"

/**
 * @brief Projects 3D Gaussians to 2D screen space.
 * 
 * World space is centered at the origin with:
 * 
 * - x in [-width/2, width/2]
 * 
 * - y in [-height/2, height/2]
 * 
 * - z is assumed to be in [-1, 1]
 * 
 * To get proper perspective projection, swap this layer
 * out with a perspective project layer that applies
 * the correct non-linear transform forward and backward pass.
 */
class NDCProjectLayer
{
public:
    ~NDCProjectLayer() { free(); }

    void allocate(int count);
    void free();
    void zero_grad();

    // wiring
    void setInput(const GaussianParams *params)   { input = params; }
    void setGradOutput(const Splat2DGrads *grads) { gradOutput = grads; }
    void setGradInput(GaussianOptState *grads)    { gradInput = grads; }
    const Splat2DParams &getOutput() const        { return output; }

    void forward(int width, int height);
    void backward(int width, int height);

private:
    // not owned
    const GaussianParams *input      = nullptr;
    const Splat2DGrads   *gradOutput = nullptr;
    GaussianOptState     *gradInput  = nullptr;

    // owned
    Splat2DParams output;
    int allocatedCount = 0;
};
