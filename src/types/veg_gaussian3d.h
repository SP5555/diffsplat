#pragma once
#include <vector>
#include "../cuda/cuda_buffer.h"
#include "gpu_soa.h"

/* ===== ===== VegGaussian3D (CPU) ===== ===== */

struct VegGaussian3D
{
    float pos_x,   pos_y,   pos_z;
    float scale_x, scale_y, scale_z;   // log-scale
    float rot_w,   rot_x,   rot_y,   rot_z; // unit quaternion
    float logit_opacity;  // pre-sigmoid
    float logit_scalar;   // pre-sigmoid VEG scalar attribute
};

/* ===== ===== VegGaussian3DParams (GPU SoA) ===== ===== */

struct VegGaussian3DParams : GpuSoA
{
    CudaBuffer<float> pos_x,   pos_y,   pos_z;
    CudaBuffer<float> scale_x, scale_y, scale_z;
    CudaBuffer<float> rot_w,   rot_x,   rot_y,   rot_z;
    CudaBuffer<float> logit_opacity;
    CudaBuffer<float> logit_scalar;

    std::vector<CudaBuffer<float>*> fields() override {
        return {&pos_x,   &pos_y,   &pos_z,
                &scale_x, &scale_y, &scale_z,
                &rot_w,   &rot_x,   &rot_y,   &rot_z,
                &logit_opacity, &logit_scalar};
    }
};

// Stub grad type -- VEG viewer is forward-only, no backward pass.
struct VegGaussian3DGrads : GpuSoA
{
    std::vector<CudaBuffer<float>*> fields() override { return {}; }
    void allocate(int) override {}
};
