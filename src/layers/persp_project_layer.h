#pragma once
#include <glm/glm.hpp>

#include "layer.h"
#include "../types/splat3d.h"
#include "../types/splat2d.h"

/**
 * @brief Projects Splat3D in world space to Splat2D in NDC/screen space
 * using a full perspective camera transform.
 * 
 * Position:
 *   clip = PV * [x y z 1]^T
 *   NDC  = clip.xyz / clip.w
 * 
 * Covariance (2x3 Jacobian of NDC w.r.t. world pos, derived from PV):
 *   J[0] = (PV[col0]*c.w - PV[col3]*c.x) / c.w^2
 *   J[1] = (PV[col1]*c.w - PV[col3]*c.y) / c.w^2
 *   Cov_2D = J * Cov_3D * J^T
 * 
 * Call setCamera() every frame before forward().
 * Backward pass is differentiable w.r.t. Splat3D params (not camera).
 */
class PerspProjectLayer : public Layer
{
public:
    ~PerspProjectLayer() {}

    void allocate(int count);
    void forward()      override;
    void backward()     override;
    void zero_grad()    override;

    // call every frame with updated camera matrices
    void setCamera(const glm::mat4 &view, const glm::mat4 &proj);

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

    /* ---- camera (set every frame, uploaded to device) ---- */
    float h_pv[16];        // PV = P * V, row-major on device
    float *d_pv = nullptr; // device copy

    /* ---- config ---- */
    int allocatedCount = 0;
};
