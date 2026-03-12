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

    std::cout << "[GaussPlyViewer] Init " << w << "x" << h
              << " tiles=" << NUM_TILES_X << "x" << NUM_TILES_Y
              << " maxPairs=" << maxPairs() << "\n";
}

void GaussPlyViewer::loadPLY(const std::string &path, const float sceneScale)
{
    auto splats = PLYLoader::load(path);
    if (splats.empty())
        throw std::runtime_error("[GaussPlyViewer] PLY loaded 0 splats: " + path);

    normalizeSplats(splats, sceneScale);

    gaussianParams.upload(splats);
}

const float *GaussPlyViewer::getOutput() const
{
    return rasLayer.getOutput();
}

/* ===== ===== Normalization ===== ===== */

void GaussPlyViewer::normalizeSplats(std::vector<Gaussian3D> &splats, const float sceneScale)
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

        // OpenCV -> OpenGL convention
        // flip along any one axis to fix left-handedness
        // here we flip Z. Flipping Y would make the world upside down.
        g.z      = -g.z;
        g.rot_w  = -g.rot_w;
        g.rot_z  = -g.rot_z;

        maxExt = std::max(maxExt, std::abs(g.x));
        maxExt = std::max(maxExt, std::abs(g.y));
        maxExt = std::max(maxExt, std::abs(g.z));
    }

    // scale to [-1, 1]
    if (maxExt > 0.f)
    {
        float linearScale = sceneScale / maxExt;
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

/* ===== ===== Cleanup ===== ===== */

void GaussPlyViewer::free()
{
    gaussianParams.free();
    activLayer.free();
    perspLayer.free();
    rasLayer.free();
}
