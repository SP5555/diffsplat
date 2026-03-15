#include "gauss_ply_viewer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "../loaders/ply_loader.h"
#include "../utils/cuda_utils.h"
#include "../utils/splat_utils.h"

/* ===== ===== Lifecycle ===== ===== */

void GaussPlyViewer::init(int w, int h)
{
    width       = w;
    height      = h;

    std::cout << "[GaussPlyViewer] Init " << w << "x" << h
              << " tiles=" << NUM_TILES_X << "x" << NUM_TILES_Y
              << " maxPairs=" << maxPairs() << "\n";
}

void GaussPlyViewer::loadPLY(const std::string &path, const float sceneScale)
{
    auto splats = PLYLoader::load(path);
    if (splats.empty())
        throw std::runtime_error("[GaussPlyViewer] PLY loaded 0 splats: " + path);

    SplatUtils::normalizeScene(splats, sceneScale);
    gaussian_params.upload(splats);
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
    atv_layer.allocate(count);
    psp_layer.allocate(count);
    ras_layer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, maxPairs(), count);

    // wire forward
    atv_layer.setInput(&gaussian_params);
    psp_layer.setInput(&atv_layer.getOutput());
    ras_layer.setInput(&psp_layer.getOutput());

    // no backward wiring, forward-only pipeline

    // register in pipeline
    pipeline.add(&atv_layer);
    pipeline.add(&psp_layer);
    pipeline.add(&ras_layer);
}

/* ===== ===== Render ===== ===== */

void GaussPlyViewer::render(const glm::mat4 &view, const glm::mat4 &proj)
{
    psp_layer.setCamera(view, proj);

    pipeline.forward();
}

void GaussPlyViewer::resize(int newWidth, int newHeight)
{
    width  = newWidth;
    height = newHeight;

    ras_layer.resize(width, height);
}
