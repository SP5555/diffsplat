#pragma once
#include "../types/gaussian3d.h"
#include "../types/splat3d.h"

/**
 * @brief Computes 3D Gaussian covariance matrices from scale and rotation.
 *
 * Forward pass:
 *   Given log-scale s and quaternion q = (w, x, y, z) (normalized in kernel):
 *
 *   S = diag(exp(s_x), exp(s_y), exp(s_z))
 *   R = rotation matrix from normalized q
 *   Cov = R * S * S^T * R^T
 *
 *   Position, color, opacity are copied through unchanged.
 *
 * Backward pass:
 *   Receives dL/dCov from NDCProjectLayer via ownedGradInput.
 *   Chains gradients back to log-scale and quaternion in Gaussian3DOptState,
 *   including the Jacobian of L2 quaternion normalization.
 */
class CovarianceLayer
{
public:
    ~CovarianceLayer() { free(); }

    void allocate(int count);
    void free();
    void zero_grad();

    // wiring
    void setInput(const Gaussian3DParams *params) { input = params; }
    const Splat3DParams &getOutput()   const      { return output; }
    void setGradOutput(const Splat3DGrads *grads) { gradOutput = grads; }
    const Gaussian3DOptState &getGradInput() const { return gradInput; }

    void forward();
    void backward();

private:
    /* ---- forward input (not owned) ---- */
    const Gaussian3DParams *input = nullptr;

    /* ---- forward output (owned) ---- */
    Splat3DParams output;

    /* ---- backward input (not owned) ---- */
    const Splat3DGrads *gradOutput = nullptr;

    /* ---- backward output (not owned) ---- */
    Gaussian3DOptState gradInput;

    /* ---- config ---- */
    int allocatedCount = 0;
};
