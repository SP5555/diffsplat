// tests/grad_check.cu
//
// Numerical gradient checker for the full diffsplat pipeline.
//
// For every parameter field in Gaussian3DParams, the analytic gradient
// produced by backward() is compared against a central-difference estimate:
//
//   numeric_grad[i] = (loss(p[i] + eps) - loss(p[i] - eps)) / (2 * eps)
//
// A relative error above REL_TOL is reported as FAIL.
// Exit code: 0 = all pass, 1 = any failure.

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda/cuda_check.h"
#include "layers/gauss_activ_layer.h"
#include "layers/tile_rasterize_layer.h"
#include "layers/mse_loss_layer.h"
#include "layers/persp_project_layer.h"
#include "pipelines/pipeline.h"
#include "types/gaussian3d.h"
#include "utils/gaussian3d_io.h"

// ===== Config =====

static constexpr int   N_SPLATS = 8;     // Gaussians in the test scene
static constexpr int   IMG_W    = 32;    // image width  (pixels)
static constexpr int   IMG_H    = 32;    // image height (pixels)
static constexpr float EPS      = 1e-3f; // finite-difference step
static constexpr float REL_TOL  = 0.05f; // max accepted relative error (5%)

// ===== Helpers =====

static float dlScalar(const CudaBuffer<float>& buf, int i)
{
    float v;
    CUDA_CHECK(cudaMemcpy(&v, buf.ptr + i, sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

static void ulScalar(CudaBuffer<float>& buf, int i, float v)
{
    CUDA_CHECK(cudaMemcpy(buf.ptr + i, &v, sizeof(float), cudaMemcpyHostToDevice));
}

static std::vector<float> dlBuffer(const CudaBuffer<float>& buf)
{
    std::vector<float> h(buf.size);
    CUDA_CHECK(cudaMemcpy(h.data(), buf.ptr, buf.size * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// ===== Scene =====

struct Scene {
    Gaussian3DParams     params;
    CudaBuffer<float>    d_target;
    GaussActivLayer      atv;
    PerspProjectLayer    psp;
    TileRasterizeLayer ras;
    MSELossLayer         mse;
    Pipeline             pipeline;
};

static void buildScene(Scene& s)
{
    std::vector<Gaussian3D> splats(N_SPLATS);
    for (int i = 0; i < N_SPLATS; i++) {
        auto& g = splats[i];
        // 2x4 grid in world space; staggered depths so z ordering is non-trivial
        g.pos_x  =  (i % 4) * 4.f - 6.f;
        g.pos_y  =  (i / 4) * 6.f - 3.f;
        g.pos_z  =  i * 0.1f;
        // Anisotropic scales ensure rotation gradients are non-zero
        g.scale_x =  1.2f;
        g.scale_y =  0.8f;
        g.scale_z =  0.5f;
        // Small tilt around Z (quaternion for ~15 degrees) so rot grads are non-zero
        g.rot_w  =  0.991f;
        g.rot_x  =  0.0f;
        g.rot_y  =  0.0f;
        g.rot_z  =  0.131f;
        g.sh_dc_r = g.sh_dc_g = g.sh_dc_b = 0.f; // SH_C0 * 0 + 0.5 bias -> color ~0.5
        g.logit_opacity = 2.f;                     // sigmoid(2) ~= 0.88
    }
    uploadGaussians(s.params, splats, 0);

    // Solid target brighter than the splat color to drive non-zero gradients
    std::vector<float> target(IMG_W * IMG_H * 3, 0.7f);
    s.d_target.allocate(IMG_W * IMG_H * 3);
    CUDA_CHECK(cudaMemcpy(s.d_target.ptr, target.data(),
                          IMG_W * IMG_H * 3 * sizeof(float), cudaMemcpyHostToDevice));

    int count = s.params.count;
    s.atv.allocate(count);
    s.psp.allocate(count);
    s.ras.allocate(IMG_W, IMG_H, count);
    s.mse.allocate(IMG_W, IMG_H);
    s.atv.allocateGrad(count);
    s.psp.allocateGrad(count);
    s.ras.allocateGrad(count);

    // Orthographic camera matching ImageFitter convention
    float hw = IMG_W * 0.5f, hh = IMG_H * 0.5f;
    float zr = std::min(hw, hh) * 0.25f;
    s.psp.setCamera(glm::mat4(1.f), glm::ortho(-hw, hw, -hh, hh, -zr, zr));
    s.atv.setCameraPosition({0.f, 0.f, 5.f});

    // Wire forward
    s.atv.setInput(&s.params);
    s.psp.setInput(&s.atv.getOutput());
    s.ras.setInput(&s.psp.getOutput());
    s.mse.setInput(s.ras.getOutput());
    s.mse.setTarget(s.d_target.ptr);

    // Wire backward
    s.ras.setGradOutput(&s.mse.getGradInput());
    s.psp.setGradOutput(&s.ras.getGradInput());
    s.atv.setGradOutput(&s.psp.getGradInput());

    s.pipeline.add(&s.atv);
    s.pipeline.add(&s.psp);
    s.pipeline.add(&s.ras);
    s.pipeline.add(&s.mse);
}

// ===== Grad check =====

static float fwdLoss(Scene& s)
{
    s.pipeline.forward();
    return s.mse.getLoss();
}

// Returns number of failures for one parameter buffer.
static int checkBuffer(
    Scene&                    s,
    const char*               name,
    CudaBuffer<float>&        param,
    const std::vector<float>& analytic)
{
    int n    = (int)param.size;
    int fail = 0;
    for (int i = 0; i < n; i++) {
        float orig = dlScalar(param, i);

        ulScalar(param, i, orig + EPS);
        float lp = fwdLoss(s);

        ulScalar(param, i, orig - EPS);
        float lm = fwdLoss(s);

        ulScalar(param, i, orig); // restore

        float num     = (lp - lm) / (2.f * EPS);
        float ag      = analytic[i];
        float denom   = std::max({std::abs(num), std::abs(ag), 1e-4f});
        float rel_err = std::abs(num - ag) / denom;

        if (rel_err > REL_TOL) {
            printf("    FAIL [%d]  analytic=%+.5f  numeric=%+.5f  rel_err=%.3f\n",
                   i, ag, num, rel_err);
            fail++;
        }
    }
    printf("  %-18s  %s  (%d/%d)\n", name, fail == 0 ? "PASS" : "FAIL", n - fail, n);
    return fail;
}

// ===== Main =====

int main()
{
    printf("=== diffsplat grad check  (%d splats, %dx%d) ===\n\n",
           N_SPLATS, IMG_W, IMG_H);

    Scene s;
    buildScene(s);

    // One analytic forward + backward to get all gradients at once
    s.pipeline.zeroGrad();
    s.pipeline.forward();
    printf("initial loss = %.6f\n\n", s.mse.getLoss());
    s.pipeline.backward();

    // Download all analytic gradients before numeric perturbations modify the scene
    auto& ag      = s.atv.getGradInput();
    auto h_pos_x  = dlBuffer(ag.grad_pos_x);
    auto h_pos_y  = dlBuffer(ag.grad_pos_y);
    auto h_pos_z  = dlBuffer(ag.grad_pos_z);
    auto h_scl_x  = dlBuffer(ag.grad_scale_x);
    auto h_scl_y  = dlBuffer(ag.grad_scale_y);
    auto h_scl_z  = dlBuffer(ag.grad_scale_z);
    auto h_rot_w  = dlBuffer(ag.grad_rot_w);
    auto h_rot_x  = dlBuffer(ag.grad_rot_x);
    auto h_rot_y  = dlBuffer(ag.grad_rot_y);
    auto h_rot_z  = dlBuffer(ag.grad_rot_z);
    auto h_sh_r   = dlBuffer(ag.grad_sh_dc_r);
    auto h_sh_g   = dlBuffer(ag.grad_sh_dc_g);
    auto h_sh_b   = dlBuffer(ag.grad_sh_dc_b);
    auto h_opac   = dlBuffer(ag.grad_logit_opacity);

    // Numeric checks -- forward-only, no backward needed per perturbation
    int fails = 0;
    fails += checkBuffer(s, "pos_x",         s.params.pos_x,         h_pos_x);
    fails += checkBuffer(s, "pos_y",         s.params.pos_y,         h_pos_y);
    fails += checkBuffer(s, "pos_z",         s.params.pos_z,         h_pos_z);
    fails += checkBuffer(s, "scale_x",       s.params.scale_x,       h_scl_x);
    fails += checkBuffer(s, "scale_y",       s.params.scale_y,       h_scl_y);
    fails += checkBuffer(s, "scale_z",       s.params.scale_z,       h_scl_z);
    fails += checkBuffer(s, "rot_w",         s.params.rot_w,         h_rot_w);
    fails += checkBuffer(s, "rot_x",         s.params.rot_x,         h_rot_x);
    fails += checkBuffer(s, "rot_y",         s.params.rot_y,         h_rot_y);
    fails += checkBuffer(s, "rot_z",         s.params.rot_z,         h_rot_z);
    fails += checkBuffer(s, "sh_dc_r",       s.params.sh_dc_r,       h_sh_r);
    fails += checkBuffer(s, "sh_dc_g",       s.params.sh_dc_g,       h_sh_g);
    fails += checkBuffer(s, "sh_dc_b",       s.params.sh_dc_b,       h_sh_b);
    fails += checkBuffer(s, "logit_opacity", s.params.logit_opacity, h_opac);

    printf("\n%s  (%d total failures)\n",
           fails == 0 ? "ALL PASS" : "SOME FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
