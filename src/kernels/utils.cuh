#pragma once

__global__ void clampFKernel(
    float* data,
    float min_val,
    float max_val,
    int n
);

void launchClampF(
    float* data,
    float min_val,
    float max_val,
    int n
);