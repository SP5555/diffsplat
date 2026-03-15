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
    void setInput(const Gaussian3DParams *params) { input = params; }
    Splat3DParams &getOutput()                    { return output; }
    void setGradOutput(const Splat3DGrads *grads) { gradOutput = grads; }
    Gaussian3DGrads &getGradInput()               { return gradInput; }

private:
    /* ---- forward input (not owned) ---- */
    const Gaussian3DParams *input = nullptr;

    /* ---- forward output (owned) ---- */
    Splat3DParams output;

    /* ---- backward input (not owned) ---- */
    const Splat3DGrads *gradOutput = nullptr;

    /* ---- backward output (owned) ---- */
    Gaussian3DGrads gradInput;

    /* ---- config ---- */
    int allocatedCount = 0;
};
