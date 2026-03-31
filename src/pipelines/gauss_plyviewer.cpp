#include "gauss_plyviewer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../cuda/cuda_check.h"
#include "../io/ply_loader.h"
#include "../utils/logs.h"
#include "../utils/splat_utils.h"

/* ===== ===== Lifecycle ===== ===== */

void GaussPlyViewer::init(int w, int h)
{
    width       = w;
    height      = h;

    log_info("GaussPlyViewer",
        "WindowSize=" + std::to_string(w) + "x" + std::to_string(h) +
        " Tiles=" + std::to_string(NUM_TILES_X) + "x" + std::to_string(NUM_TILES_Y) +
        " MaxPairs=" + std::to_string(getMaxPairs())
    );
}

void GaussPlyViewer::loadPLY(const std::string &path, const float sceneScale)
{
    auto result = PLYLoader::load(path);
    if (result.splats.empty())
        throw std::runtime_error("[GaussPlyViewer] PLY loaded 0 splats: " + path);

    SplatUtils::normalizeScene(result.splats, sceneScale);
    sh_degree = result.sh_degree;
    gaussian_params.upload(result.splats, sh_degree);
}

float *GaussPlyViewer::getOutput()
{
    return ras_layer.getOutput();
}

uint32_t GaussPlyViewer::getVisibleCount()
{
    return ras_layer.getVisibleCount();
}

/* ===== ===== Init ===== ===== */

void GaussPlyViewer::initLayers()
{
    int count = gaussian_params.count;

    // allocate
    atv_layer.setSHDegree(sh_degree);
    atv_layer.allocate(count);
    psp_layer.allocate(count);
    ras_layer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, getMaxPairs(), count);

    // wire forward
    atv_layer.setInput(&gaussian_params);
    psp_layer.setInput(&atv_layer.getOutput());
    ras_layer.setInput(&psp_layer.getOutput());

    // no backward wiring, forward-only pipeline

    // register in pipeline
    pipeline.add(&atv_layer);
    pipeline.add(&psp_layer);
    pipeline.add(&ras_layer);

    loaded = true;
}

void GaussPlyViewer::reloadPLY(const std::string &path, float sceneScale)
{
    pipeline.clear();
    loadPLY(path, sceneScale);
    initLayers();
}

/* ===== ===== Render ===== ===== */

void GaussPlyViewer::render(const glm::mat4 &view, const glm::mat4 &proj, const glm::vec3 &cam_pos)
{
    atv_layer.setCameraPosition(cam_pos);
    psp_layer.setCamera(view, proj);

    pipeline.forward();
}

void GaussPlyViewer::resize(int newWidth, int newHeight)
{
    width  = newWidth;
    height = newHeight;

    ras_layer.resize(width, height);
}
