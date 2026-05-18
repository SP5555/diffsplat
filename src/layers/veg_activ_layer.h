#pragma once
#include "layer.h"
#include "../types/veg_gaussian3d.h"
#include "../types/splat3d.h"
#include "../cuda/cuda_buffer.h"

static constexpr int VEG_LUT_SIZE = 256;

/**
 * @brief Converts VegGaussian3DParams to Splat3DParams.
 *
 * Geometry: identical to GaussActivLayer -- covariance from exp(scale) and normalized quaternion.
 * Color: sigmoid(logit_scalar) -> linear interp in a 256-entry RGB LUT.
 * Opacity: sigmoid(logit_opacity) * lut_alpha (product of learned weight and TF opacity).
 *
 * LUT lives in device memory. Call setLUT() (or initDefaultLUT()) before the first forward().
 * The VEG viewer is forward-only; backward() is a no-op.
 */
class VegActivLayer
    : public TypedLayer<VegGaussian3DParams, Splat3DParams, VegGaussian3DGrads, Splat3DGrads>
{
public:
    ~VegActivLayer() {}

    void forward()  override;
    void backward() override {}

    // Allocates output + LUT device buffers and seeds a grayscale default LUT.
    void allocate(int n) override;

    // Upload a 256-entry LUT from CPU. rgb must be [VEG_LUT_SIZE*3], alpha [VEG_LUT_SIZE].
    // Safe to call every frame from the TF widget callback.
    void setLUT(const float *rgb, const float *alpha);

    // Seed a greyscale ramp so the scene is visible before any TF interaction.
    void initDefaultLUT();

private:
    CudaBuffer<float> d_lut_rgb;   // VEG_LUT_SIZE * 3
    CudaBuffer<float> d_lut_alpha; // VEG_LUT_SIZE
};
