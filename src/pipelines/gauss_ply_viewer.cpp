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
    gaussianParams.upload(splats);
}

float *GaussPlyViewer::getOutput()
{
    return rasLayer.getOutput();
}

/* ===== ===== Init ===== ===== */

void GaussPlyViewer::initLayers()
{
    int count = gaussianParams.count;

    // allocate
    activLayer.allocate(count);
    perspLayer.allocate(count);
    rasLayer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, maxPairs(), count);

    // wire forward
    activLayer.setInput(&gaussianParams);
    perspLayer.setInput(&activLayer.getOutput());
    rasLayer.setInput(&perspLayer.getOutput());

    // no backward wiring, forward-only pipeline

    // register in pipeline
    pipeline.add(&activLayer);
    pipeline.add(&perspLayer);
    pipeline.add(&rasLayer);
}

/* ===== ===== Render ===== ===== */

void GaussPlyViewer::render(const glm::mat4 &view, const glm::mat4 &proj)
{
    perspLayer.setCamera(view, proj);

    pipeline.forward();
}

void GaussPlyViewer::resize(int newWidth, int newHeight)
{
    width  = newWidth;
    height = newHeight;

    rasLayer.resize(width, height);
}
