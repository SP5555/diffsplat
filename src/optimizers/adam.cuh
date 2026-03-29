#pragma once
#include "../types/gaussian3d.h"

#define EPSILON 1e-8f

struct AdamConfig
{
    float lr_master     = 1e-3f;

    // different parameters may require different learning rates
    // because of different scalings. manually tune them!
    float lr_pos        = lr_master * 16.f;
    float lr_scale      = lr_master * 0.4f;
    float lr_rot        = lr_master * 0.4f;
    float lr_color      = lr_master * 6.f;
    float lr_sh_rest    = lr_master * 6.f;
    float lr_opacity    = lr_master * 6.f;

    float beta1         = 0.9f;
    float beta2         = 0.999f;
    float epsilon       = EPSILON;
};


/**
 * @brief GPU-side optimization state: gradient buffers and Adam moments.
 */
struct AdamState
{
    int count = 0;

    // Adam first moments
    CudaBuffer<float> m_pos_x,   m_pos_y,   m_pos_z;
    CudaBuffer<float> m_scale_x, m_scale_y, m_scale_z;
    CudaBuffer<float> m_rot_w,   m_rot_x,   m_rot_y,   m_rot_z;
    CudaBuffer<float> m_color_r, m_color_g, m_color_b;
    CudaBuffer<float> m_sh_rest_r, m_sh_rest_g, m_sh_rest_b;
    CudaBuffer<float> m_opacity;

    // Adam second moments
    CudaBuffer<float> v_pos_x,   v_pos_y,   v_pos_z;
    CudaBuffer<float> v_scale_x, v_scale_y, v_scale_z;
    CudaBuffer<float> v_rot_w,   v_rot_x,   v_rot_y,   v_rot_z;
    CudaBuffer<float> v_color_r, v_color_g, v_color_b;
    CudaBuffer<float> v_sh_rest_r, v_sh_rest_g, v_sh_rest_b;
    CudaBuffer<float> v_opacity;

    int sh_num_bands = 0;

    void allocate(int n, int sh_degree = 0)
    {
        m_pos_x.allocate(n);   v_pos_x.allocate(n);
        m_pos_y.allocate(n);   v_pos_y.allocate(n);
        m_pos_z.allocate(n);   v_pos_z.allocate(n);
        m_scale_x.allocate(n); v_scale_x.allocate(n);
        m_scale_y.allocate(n); v_scale_y.allocate(n);
        m_scale_z.allocate(n); v_scale_z.allocate(n);
        m_rot_w.allocate(n);   v_rot_w.allocate(n);
        m_rot_x.allocate(n);   v_rot_x.allocate(n);
        m_rot_y.allocate(n);   v_rot_y.allocate(n);
        m_rot_z.allocate(n);   v_rot_z.allocate(n);
        m_color_r.allocate(n); v_color_r.allocate(n);
        m_color_g.allocate(n); v_color_g.allocate(n);
        m_color_b.allocate(n); v_color_b.allocate(n);
        m_opacity.allocate(n); v_opacity.allocate(n);

        sh_num_bands = sh_degree_to_bands(sh_degree);
        if (sh_num_bands > 0) {
            m_sh_rest_r.allocate(n * sh_num_bands); v_sh_rest_r.allocate(n * sh_num_bands);
            m_sh_rest_g.allocate(n * sh_num_bands); v_sh_rest_g.allocate(n * sh_num_bands);
            m_sh_rest_b.allocate(n * sh_num_bands); v_sh_rest_b.allocate(n * sh_num_bands);
        }

        count = n;
    }
};

/**
 * @brief Performs the Adam optimization step.
 * 
 * @param[in] gaussians         Current Gaussian parameters
 * @param[in] grads             Gradients for all Gaussian parameters
 * @param[out] opt_state        Output gradients for all Gaussian parameters
 * @param[in] config            Adam optimization configuration
 * @param[in] step              Current optimization step number (starting from 1)
 */
void launchAdam(
    Gaussian3DParams &gaussians,
    Gaussian3DGrads  &grads,
    AdamState        &opt_state,
    const AdamConfig &config,
    int step
);

class Adam
{
public:
    void init(int count, int sh_degree = 0, const AdamConfig &cfg = {})
    {
        config = cfg;
        state.allocate(count, sh_degree);
    }
    void step(Gaussian3DParams &gaussians, Gaussian3DGrads  &grads)
    {
        launchAdam(gaussians, grads, state, config, ++step_count);
    }
    int getStepCount()            const { return step_count; }
    const AdamConfig &getConfig() const { return config; }

private:
    AdamConfig config;
    AdamState  state;
    int        step_count = 0;
};