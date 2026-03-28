#include "adam.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

/**
 * @brief Adam optimizer kernel for updating Gaussian parameters on the GPU.
 * 
 * @param[in, out] param    Parameter array to be updated
 * @param[in] grad          Gradient array computed from the backward pass
 * @param[in, out] moment   First moment (mean) array for Adam
 * @param[in, out] variance Second moment (variance) array for Adam
 * @param[in] lr            Learning rate
 * @param[in] beta1         Exponential decay rate for the first moment
 * @param[in] beta2         Exponential decay rate for the second moment
 * @param[in] epsilon       Small constant for numerical stability
 * @param[in] bc1           Bias correction factor for the first moment
 * @param[in] bc2           Bias correction factor for the second moment
 * @param[in] n             Number of parameters
 */
__global__ void adamKernel(
    float* param,
    const float* grad,
    float* moment,
    float* variance,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bc1,
    float bc2,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];

    // update first and second moments
    float m = moment[idx] = beta1 * moment[idx] + (1.f - beta1) * g;
    float v = variance[idx] = beta2 * variance[idx] + (1.f - beta2) * g * g;

    // bias correction
    float m_hat = m * bc1;
    float v_hat = v * bc2;

    // update parameter
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

/**
 * Helper function to launch the Adam optimizer kernel
 * for a single parameter array. 
 */
static void stepOne(
    float* param,
    const float* grad,
    float* moment,
    float* variance,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bc1,
    float bc2,
    int n)
{
    int blocks = (n + BLOCK_SIZE- 1) / BLOCK_SIZE;
    adamKernel<<<blocks, BLOCK_SIZE>>>(
        param,
        grad,
        moment,
        variance,
        lr,
        beta1,
        beta2,
        epsilon,
        bc1,
        bc2,
        n
    );
}

void launchAdam(
    Gaussian3DParams &gaussians,
    Gaussian3DGrads  &grads,
    AdamState        &opt_state,
    const AdamConfig &config,
    int step)
{
    int n = gaussians.count;
    float bc1 = 1.f / (1.f - powf(config.beta1, step));
    float bc2 = 1.f / (1.f - powf(config.beta2, step));
    
    auto go = [&](
        CudaBuffer<float>& p,
        const CudaBuffer<float>& g,
        CudaBuffer<float>& m,
        CudaBuffer<float>& v,
        float lr) {
        stepOne(
            p, g, m, v, lr,
            config.beta1, config.beta2, config.epsilon,
            bc1, bc2, n
        );
    };
    
    // brevity
    auto& g = gaussians;
    auto& gr = grads;
    auto& o = opt_state;
    auto& c = config;
    go(g.pos_x,   gr.grad_pos_x,   o.m_pos_x,   o.v_pos_x,   c.lr_pos);
    go(g.pos_y,   gr.grad_pos_y,   o.m_pos_y,   o.v_pos_y,   c.lr_pos);
    go(g.pos_z,   gr.grad_pos_z,   o.m_pos_z,   o.v_pos_z,   c.lr_pos);
    go(g.scale_x, gr.grad_scale_x, o.m_scale_x, o.v_scale_x, c.lr_scale);
    go(g.scale_y, gr.grad_scale_y, o.m_scale_y, o.v_scale_y, c.lr_scale);
    go(g.scale_z, gr.grad_scale_z, o.m_scale_z, o.v_scale_z, c.lr_scale);
    go(g.rot_w,   gr.grad_rot_w,   o.m_rot_w,   o.v_rot_w,   c.lr_rot);
    go(g.rot_x,   gr.grad_rot_x,   o.m_rot_x,   o.v_rot_x,   c.lr_rot);
    go(g.rot_y,   gr.grad_rot_y,   o.m_rot_y,   o.v_rot_y,   c.lr_rot);
    go(g.rot_z,   gr.grad_rot_z,   o.m_rot_z,   o.v_rot_z,   c.lr_rot);
    go(g.color_sh_r, gr.grad_color_sh_r, o.m_color_r, o.v_color_r, c.lr_color);
    go(g.color_sh_g, gr.grad_color_sh_g, o.m_color_g, o.v_color_g, c.lr_color);
    go(g.color_sh_b, gr.grad_color_sh_b, o.m_color_b, o.v_color_b, c.lr_color);
    go(g.logit_opacity, gr.grad_logit_opacity, o.m_opacity, o.v_opacity, c.lr_opacity);
    CUDA_SYNC_CHECK();
}