#include "gauss_ply_viewer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "../loaders/ply_loader.h"
#include "../utils/cuda_utils.cuh"

/* ===== ===== Lifecycle ===== ===== */

GaussPlyViewer::~GaussPlyViewer()
{
    free();
}

void GaussPlyViewer::init(int w, int h)
{
    width       = w;
    height      = h;
    NUM_TILES_X = (w + TILE_SIZE - 1) / TILE_SIZE;
    NUM_TILES_Y = (h + TILE_SIZE - 1) / TILE_SIZE;

    std::cout << "[GaussPlyViewer] Init " << w << "x" << h
              << " tiles=" << NUM_TILES_X << "x" << NUM_TILES_Y
              << " maxPairs=" << maxPairs() << "\n";
}

void GaussPlyViewer::loadPLY(const std::string &path)
{
    auto splats = PLYLoader::load(path);
    if (splats.empty())
        throw std::runtime_error("[GaussPlyViewer] PLY loaded 0 splats: " + path);

    normalizeSplats(splats);

    gaussianParams.upload(splats);
    initLayers();
}

const float *GaussPlyViewer::getOutput() const
{
    return rasLayer.getOutput();
}

/* ===== ===== Normalization ===== ===== */

void GaussPlyViewer::normalizeSplats(std::vector<Gaussian3D> &splats)
{
    // compute centroid
    float cx = 0.f, cy = 0.f, cz = 0.f;
    for (const auto &g : splats)
    {
        cx += g.x;
        cy += g.y;
        cz += g.z;
    }
    float inv = 1.f / (float)splats.size();
    cx *= inv; cy *= inv; cz *= inv;

    // translate to centroid, find max extent
    float maxExt = 0.f;
    for (auto &g : splats)
    {
        g.x -= cx; g.y -= cy; g.z -= cz;
        maxExt = std::max(maxExt, std::abs(g.x));
        maxExt = std::max(maxExt, std::abs(g.y));
        maxExt = std::max(maxExt, std::abs(g.z));
    }

    // scale to [-1, 1]
    if (maxExt > 0.f)
    {
        float linearScale = 1.f / maxExt;
        float logScale = logf(linearScale);
        for (auto &g : splats)
        {
            g.x *= linearScale;
            g.y *= linearScale;
            g.z *= linearScale;

            g.scale_x += logScale;
            g.scale_y += logScale;
            g.scale_z += logScale;
        }
    }
}

/* ===== ===== Init ===== ===== */

void GaussPlyViewer::initLayers()
{
    int count = gaussianParams.count;

    // allocate
    activLayer.allocate(count);
    // perspLayer.allocate(width, height, count);  // TODO
    rasLayer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, maxPairs(), count);

    // wire forward
    // activLayer.setInput(&gaussianParams);
    // perspLayer.setInput(&activLayer.getOutput());  // TODO
    // rasLayer.setInput(&perspLayer.getOutput());    // TODO
    // rasLayer.setInput(&activLayer.getOutput());

    // no backward wiring, forward-only pipeline
}

/* ===== ===== Render ===== ===== */

void GaussPlyViewer::render(/* const glm::mat4 &view, const glm::mat4 &proj */)
{
    // activLayer.forward();
    // perspLayer.setCamera(view, proj);  // TODO
    // perspLayer.forward();              // TODO
    // rasLayer.forward();
}

/* ===== ===== Cleanup ===== ===== */

void GaussPlyViewer::free()
{
    gaussianParams.free();
    activLayer.free();
    // perspLayer.free();  // TODO
    rasLayer.free();
}
