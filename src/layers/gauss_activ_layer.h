#pragma once

#include "layer.h"
#include "../types/gaussian3d.h"
#include "../types/splat3d.h"

/**
 * @brief Computes 3D Gaussian covariance matrices from scale and rotation.
 *
 * Forward pass: Given log-scale s and quaternion q = (w, x, y, z) (normalized in kernel):
 *
 *   S = diag(exp(s_x), exp(s_y), exp(s_z))
 *   R = rotation matrix from normalized q
 *   Cov = R * S * S^T * R^T
 *
 *   Position, color, opacity are copied through unchanged.
 *
 * Backward pass: Receives gradients of 2D slpat parameters and computes
 * gradients w.r.t. 3D splat parameters.
 */
class GaussActivLayer : public Layer
{
public:
    ~GaussActivLayer() {}

    void allocate(int count);
    void forward()      override;
    void backward()     override;
    void zero_grad()    override;

    // wiring
    void setInput(const Gaussian3DParams *params) { in = params; }
    Splat3DParams &getOutput()                    { return out; }
    void setGradOutput(const Splat3DGrads *grads) { grad_out = grads; }
    Gaussian3DGrads &getGradInput()               { return grad_in; }

private:
    /* ---- forward input (not owned) ---- */
    const Gaussian3DParams *in = nullptr;

    /* ---- forward output (owned) ---- */
    Splat3DParams out;

    /* ---- backward input (not owned) ---- */
    const Splat3DGrads *grad_out = nullptr;

    /* ---- backward output (owned) ---- */
    Gaussian3DGrads grad_in;

    /* ---- config ---- */
    int allocated_count = 0;
};
