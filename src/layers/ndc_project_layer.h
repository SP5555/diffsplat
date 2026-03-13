#pragma once

#include "layer.h"
#include "../types/splat3d.h"
#include "../types/splat2d.h"
#include "../utils/cuda_utils.cuh"

/**
 * @brief Projects Splat3D in world space to Splat2D in screen space.
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
class NDCProjectLayer : public Layer
{
public:
    ~NDCProjectLayer() {}

    void allocate(int width, int height, int count);
    void forward()      override;
    void backward()     override;
    void zero_grad()    override;

    // wiring
    void setInput(const Splat3DParams *params)    { input = params; }
    Splat2DParams &getOutput()                    { return output; }
    void setGradOutput(const Splat2DGrads *grads) { gradOutput = grads; } 
    Splat3DGrads &getGradInput()                  { return gradInput; }

private:
    /* ---- forward input (not owned) ---- */
    const Splat3DParams *input = nullptr;

    /* ---- forward output (owned) ---- */
    Splat2DParams output;

    /* ---- backward input (not owned) ---- */
    const Splat2DGrads *gradOutput = nullptr;

    /* ---- backward output (owned) ---- */
    Splat3DGrads gradInput;

    /* ---- config ---- */
    int screen_width  = 0;
    int screen_height = 0;
    int allocatedCount = 0;
};