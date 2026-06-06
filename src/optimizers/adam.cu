#include "adam.h"

#include <cuda_runtime.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

/**
 * @brief Adam update kernel for a single parameter array.
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

    float m = moment[idx]   = beta1 * moment[idx]   + (1.f - beta1) * g;
    float v = variance[idx] = beta2 * variance[idx] + (1.f - beta2) * g * g;

    float m_hat = m * bc1;
    float v_hat = v * bc2;

    param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

static __global__ void zeroSlotsKernel(float* m, float* v, const int* indices, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int slot = indices[i];
    m[slot] = 0.f;
    v[slot] = 0.f;
}

void Adam::zeroGroupMoments(int group_idx)
{
    auto& g = groups[group_idx];
    g.m.zero();
    g.v.zero();
}

void Adam::zeroSlotMoments(const int* d_indices, int count, int n_match)
{
    int blocks = divRoundUp(count, BLOCK_SIZE);
    for (auto& g : groups) {
        if (g.n != n_match) continue;
        zeroSlotsKernel<<<blocks, BLOCK_SIZE>>>(g.m, g.v, d_indices, count);
    }
    CUDA_SYNC_CHECK();
}

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
    int blocks = divRoundUp(n, BLOCK_SIZE);
    adamKernel<<<blocks, BLOCK_SIZE>>>(
        param, grad, moment, variance,
        lr, beta1, beta2, epsilon, bc1, bc2, n
    );
}

void Adam::step()
{
    ++step_count;
    float bc1 = 1.f / (1.f - powf(hp.beta1, step_count));
    float bc2 = 1.f / (1.f - powf(hp.beta2, step_count));

    for (auto& g : groups) {
        stepOne(
            g.param, g.grad, g.m, g.v,
            g.lr, hp.beta1, hp.beta2, hp.epsilon,
            bc1, bc2, g.n
        );
    }
    CUDA_SYNC_CHECK();
}
