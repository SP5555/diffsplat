#include "veg_renderer.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../cuda/cuda_check.h"
#include "../io/veg_ply_io.h"
#include "../utils/logs.h"

/* ===== ===== Lifecycle ===== ===== */

void VegRenderer::init(int w, int h)
{
    width  = w;
    height = h;
    log_info("VegRenderer", "WindowSize=" + std::to_string(w) + "x" + std::to_string(h));
}

/* ===== ===== PLY loading ===== ===== */

static void normalizeVegScene(std::vector<VegGaussian3D> &splats, float scene_scale)
{
    float cx = 0.f, cy = 0.f, cz = 0.f;
    for (const auto &g : splats) { cx += g.pos_x; cy += g.pos_y; cz += g.pos_z; }
    float inv = 1.f / (float)splats.size();
    cx *= inv; cy *= inv; cz *= inv;

    float max_ext = 0.f;
    for (auto &g : splats)
    {
        g.pos_x -= cx; g.pos_y -= cy; g.pos_z -= cz;
        max_ext = std::max({max_ext, std::abs(g.pos_x), std::abs(g.pos_y), std::abs(g.pos_z)});
    }

    if (max_ext > 0.f)
    {
        float linear_scale = scene_scale / max_ext;
        float log_scale    = logf(linear_scale);
        for (auto &g : splats)
        {
            g.pos_x   *= linear_scale;
            g.pos_y   *= linear_scale;
            g.pos_z   *= linear_scale;
            g.scale_x += log_scale;
            g.scale_y += log_scale;
            g.scale_z += log_scale;
        }
    }
}

static void uploadVegGaussians(VegGaussian3DParams &params,
                               const std::vector<VegGaussian3D> &splats)
{
    int n = (int)splats.size();
    params.allocate(n);

    std::vector<float> tmp(n);
    auto upload = [&](CudaBuffer<float> &buf, auto getter) {
        for (int i = 0; i < n; i++) tmp[i] = getter(splats[i]);
        CUDA_CHECK(cudaMemcpy(buf.ptr, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    upload(params.pos_x,         [](const VegGaussian3D &g){ return g.pos_x; });
    upload(params.pos_y,         [](const VegGaussian3D &g){ return g.pos_y; });
    upload(params.pos_z,         [](const VegGaussian3D &g){ return g.pos_z; });
    upload(params.scale_x,       [](const VegGaussian3D &g){ return g.scale_x; });
    upload(params.scale_y,       [](const VegGaussian3D &g){ return g.scale_y; });
    upload(params.scale_z,       [](const VegGaussian3D &g){ return g.scale_z; });
    upload(params.rot_w,         [](const VegGaussian3D &g){ return g.rot_w; });
    upload(params.rot_x,         [](const VegGaussian3D &g){ return g.rot_x; });
    upload(params.rot_y,         [](const VegGaussian3D &g){ return g.rot_y; });
    upload(params.rot_z,         [](const VegGaussian3D &g){ return g.rot_z; });
    upload(params.logit_opacity, [](const VegGaussian3D &g){ return g.logit_opacity; });
    upload(params.logit_scalar,  [](const VegGaussian3D &g){ return g.logit_scalar; });
}

void VegRenderer::loadPLY(const std::string &path, float sceneScale)
{
    auto splats = VegPLYLoader::load(path);
    if (splats.empty())
        throw std::runtime_error("[VegRenderer] PLY loaded 0 gaussians: " + path);

    normalizeVegScene(splats, sceneScale);
    uploadVegGaussians(veg_params, splats);
}

/* ===== ===== Layer wiring ===== ===== */

void VegRenderer::initLayers()
{
    int count = veg_params.count;

    // allocate forward buffers only (no grad buffers needed for viewer)
    veg_layer.allocate(count); // also allocates LUT + seeds default grayscale
    psp_layer.allocate(count);
    ras_layer.allocate(width, height, count);

    // wire forward
    veg_layer.setInput(&veg_params);
    psp_layer.setInput(&veg_layer.getOutput());
    ras_layer.setInput(&psp_layer.getOutput());

    // no backward wiring, forward-only pipeline

    // register in pipeline
    pipeline.add(&veg_layer);
    pipeline.add(&psp_layer);
    pipeline.add(&ras_layer);

    loaded = true;
}

void VegRenderer::reloadPLY(const std::string &path, float sceneScale)
{
    loaded = false;
    pipeline.clear();
    loadPLY(path, sceneScale);
    initLayers();
}

/* ===== ===== Render ===== ===== */

void VegRenderer::render(const glm::mat4 &view, const glm::mat4 &proj)
{
    psp_layer.setCamera(view, proj);
    pipeline.forward();
}

void VegRenderer::resize(int newWidth, int newHeight)
{
    width  = newWidth;
    height = newHeight;
    ras_layer.resize(width, height);
}

float *VegRenderer::getOutput()
{
    return ras_layer.getOutput();
}

uint32_t VegRenderer::getVisibleCount()
{
    return ras_layer.getVisibleCount();
}
