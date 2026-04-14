#pragma once

/**
 * Base class for all differentiable layers.
 *
 * Ownership convention:
 *   Forward  : input (non-owning ptr) --> [layer] --> output (owned)
 *   Backward : grad_input (owned)     <-- [layer] <-- grad_output (non-owning ptr)
 *
 * "input" and "output" are always relative to THIS layer, not the direction
 * of flow. This means grad_input is what the layer PRODUCES during backward
 * and sends upstream -- not what it receives.  grad_output is what it
 * RECEIVES from the downstream layer to start the backward computation.
 *
 * In short:
 *   forward()  reads `in`,       writes `out`
 *   backward() reads `grad_out`, writes `grad_in`
 */
class Layer
{
public:
    virtual ~Layer() = default;
    virtual void forward()   = 0;
    virtual void backward()  = 0;
    virtual void zero_grad() = 0;
};