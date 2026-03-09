#include <cuda_runtime.h>
#include <math.h>

#include "adam.cuh"

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
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adamKernel<<<blocks, threads>>>(
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
    GaussianParams &gaussians,
    GaussianOptState &opt_state,
    const AdamConfig &config,
    int step)
{
    int n = gaussians.count;
    float bc1 = 1.f / (1.f - powf(config.beta1, step));
    float bc2 = 1.f / (1.f - powf(config.beta2, step));

    
    auto go = [&](float* p, const float* g, float* m, float* v, float lr) {
        stepOne(
            p, g, m, v, lr,
            config.beta1, config.beta2, config.epsilon,
            bc1, bc2, n
        );
    };
    
    // brevity
    auto g = gaussians;
    auto o = opt_state;
    auto c = config;
    go(g.pos_x, o.grad_pos_x, o.m_pos_x, o.v_pos_x, c.lr_pos);
    go(g.pos_y, o.grad_pos_y, o.m_pos_y, o.v_pos_y, c.lr_pos);
    go(g.cov_a, o.grad_cov_a, o.m_cov_a, o.v_cov_a, c.lr_cov);
    go(g.cov_b, o.grad_cov_b, o.m_cov_b, o.v_cov_b, c.lr_cov);
    go(g.cov_d, o.grad_cov_d, o.m_cov_d, o.v_cov_d, c.lr_cov);
    go(g.color_r, o.grad_color_r, o.m_color_r, o.v_color_r, c.lr_color);
    go(g.color_g, o.grad_color_g, o.m_color_g, o.v_color_g, c.lr_color);
    go(g.color_b, o.grad_color_b, o.m_color_b, o.v_color_b, c.lr_color);
    go(g.opacity, o.grad_opacity, o.m_opacity, o.v_opacity, c.lr_opacity);

    cudaDeviceSynchronize();
}