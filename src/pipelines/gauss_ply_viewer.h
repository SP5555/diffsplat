#pragma once
#include <string>

#include "../types/gaussian3d.h"
#include "../layers/gauss_activ_layer.h"
// #include "../layers/perspective_project_layer.h"  // TODO
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
    void loadPLY(const std::string &path);
    void initLayers();

    void render(/* const glm::mat4 &view, const glm::mat4 &proj */);

    const float *getOutput() const;

    void free();

private:
    void normalizeSplats(std::vector<Gaussian3D> &splats);

    // ---- config ----
    static constexpr int TILE_SIZE   = 16;
    int NUM_TILES_X = 0;
    int NUM_TILES_Y = 0;
    int maxPairs() const { return NUM_TILES_X * NUM_TILES_Y * 50; }

    // ---- state ----
    int width  = 0;
    int height = 0;

    // ---- data ----
    Gaussian3DParams gaussianParams;

    // ---- layers ----
    GaussActivLayer  activLayer;
    // PerspectiveProjectLayer perspLayer;  // TODO
    RasterizeLayer   rasLayer;
};
