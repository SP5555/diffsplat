#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <iostream>

/**
 * @brief CPU side structure for a single Gaussian splat.
 * 
 * Contains position, covariance, color and opacity.
 */
struct Gaussian3D
{
    float x, y, z;
    float cov_a, cov_b, cov_d;
    float r, g, b;
    float opacity;
    
    // TODO: add loader from PLY or similar in the future
};

/**
 * @brief GPU side structure for Gaussian parameters.
 * 
 * Contains position, covariance, color and opacity for each Gaussian,
 * stored in separate arrays for coalesced access.
 */
struct GaussianParams
{
    int count = 0;

    // XYZ in NDC [-1, 1]
    float *pos_x = nullptr;
    float *pos_y = nullptr;
    float *pos_z = nullptr;

    // covariance upper triangle [[a,b],[b,d]]
    float *cov_a = nullptr;
    float *cov_b = nullptr;
    float *cov_d = nullptr;

    float *color_r = nullptr;
    float *color_g = nullptr;
    float *color_b = nullptr;
    float *opacity = nullptr;

    void upload(const std::vector<Gaussian3D> &host);
    void allocateDeviceMem(int n);
    void free();

    // Random init: splats scattered in NDC [-1, 1], small covariance
    // width and height are good to have for aspect ratio correction of
    // initial splat distribution
    static GaussianParams randomInit(int n, int width, int height, int seed = 42);
};

/**
 * @brief GPU side structure for optimization state of the Gaussians.
 * 
 * Contains gradient buffers and Adam moment buffers for each parameter.
 */
struct GaussianOptState
{
    int count = 0;

    // Gradient buffers (zeroed each iteration)
    float *grad_pos_x = nullptr;
    float *grad_pos_y = nullptr;
    float *grad_cov_a = nullptr;
    float *grad_cov_b = nullptr;
    float *grad_cov_d = nullptr;
    float *grad_color_r = nullptr;
    float *grad_color_g = nullptr;
    float *grad_color_b = nullptr;
    float *grad_opacity = nullptr;

    // Adam moment buffers (zeroed once at init)
    float *m_pos_x = nullptr;
    float *m_pos_y = nullptr;
    float *m_cov_a = nullptr;
    float *m_cov_b = nullptr;
    float *m_cov_d = nullptr;
    float *m_color_r = nullptr;
    float *m_color_g = nullptr;
    float *m_color_b = nullptr;
    float *m_opacity = nullptr;
    float *v_pos_x = nullptr;
    float *v_pos_y = nullptr;
    float *v_cov_a = nullptr;
    float *v_cov_b = nullptr;
    float *v_cov_d = nullptr;
    float *v_color_r = nullptr;
    float *v_color_g = nullptr;
    float *v_color_b = nullptr;
    float *v_opacity = nullptr;

    void allocateDeviceMem(int n);
    void zero_grad();
    void free();
};