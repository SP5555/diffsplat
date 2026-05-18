#pragma once
#include <string>

#include "pipeline.h"
#include "../types/veg_gaussian3d.h"
#include "../layers/veg_activ_layer.h"
#include "../layers/persp_project_layer.h"
#include "../layers/tile_rasterize_layer.h"

/**
 * @brief Forward-only VEG splatting pipeline.
 *
 * Loads a VEG PLY file (element "gaussians" with a scalar attribute),
 * normalizes positions to [-1,1], and renders through:
 *   VegActivLayer -> PerspProjectLayer -> TileRasterizeLayer
 *
 * Call setLUT() whenever the transfer function changes; it propagates to VegActivLayer.
 */
class VegRenderer
{
public:
    ~VegRenderer() {}

    void init(int w, int h);
    void reloadPLY(const std::string &path, float sceneScale = 1.f);

    bool isLoaded() const { return loaded; }

    void render(const glm::mat4 &view, const glm::mat4 &proj);
    void resize(int newWidth, int newHeight);

    float    *getOutput();
    uint32_t  getVisibleCount();

    // Forward a 256-entry LUT from the TF widget to the activation layer.
    void setLUT(const float *rgb, const float *alpha) { veg_layer.setLUT(rgb, alpha); }

private:
    void loadPLY(const std::string &path, float sceneScale);
    void initLayers();

    int  width  = 0;
    int  height = 0;
    bool loaded = false;

    VegGaussian3DParams veg_params;

    Pipeline           pipeline;
    VegActivLayer      veg_layer;
    PerspProjectLayer  psp_layer;
    TileRasterizeLayer ras_layer;
};
