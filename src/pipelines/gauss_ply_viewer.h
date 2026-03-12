#pragma once
#include <string>

#include "../types/gaussian3d.h"
#include "../layers/gauss_activ_layer.h"
#include "../layers/persp_project_layer.h"
#include "../layers/rasterize_layer.h"

/**
 * @brief Forward-only Gaussian splatting pipeline for PLY scene viewing.
 * 
 * Loads a PLY file, normalizes splat positions to [-1, 1], and renders
 * forward-only (no backward pass, no optimizer).
 */
class GaussPlyViewer
{
public:
    ~GaussPlyViewer();

    void init(int w, int h);
    void loadPLY(const std::string &path, const float sceneScale = 1.f);
    void initLayers();

    void render(const glm::mat4 &view, const glm::mat4 &proj);
    void resize(int newWidth, int newHeight);

    const float *getOutput() const;

    void free();

private:
    void normalizeSplats(std::vector<Gaussian3D> &splats, const float sceneScale);

    // ---- config ----
    static constexpr int NUM_TILES_X = 64;
    static constexpr int NUM_TILES_Y = 64;
    int maxPairs() const { return NUM_TILES_X * NUM_TILES_Y * 4096; }

    // ---- state ----
    int width  = 0;
    int height = 0;

    // ---- data ----
    Gaussian3DParams gaussianParams;

    // ---- layers ----
    GaussActivLayer   activLayer;
    PerspProjectLayer perspLayer;
    RasterizeLayer    rasLayer;
};
