#pragma once
#include <string>
#include <cuda_runtime.h>

#include "pipeline.h"
#include "../types/gaussian3d.h"
#include "../layers/gauss_activ_layer.h"
#include "../layers/ndc_project_layer.h"
#include "../layers/rasterize_layer.h"
#include "../layers/mse_loss_layer.h"
#include "../optimizers/adam.cuh"

/**
 * @brief Differentiable Gaussian image fitter.
 * 
 * Pure CUDA pipeline, no GL, no display, no window.
 * Owns Gaussian parameters and all layers.
 * Call render() each frame, then getOutput() to get the pixel buffer
 * for display via AppBase::displayFrame().
 */
class GaussImgFitter
{
public:
    ~GaussImgFitter();

    void init(int width, int height);
    void loadTargetImage(const std::string &imagePath, int w, int h, int padding = 0);
    void randomInitGaussians(int count, int seed = -1);
    void initLayers();
    void free();

    void render();

    const float *getOutput()    const;
    int          getIterCount() const { return iterCount; }

private:
    int maxPairs() const { return powf(2.f, 20.f); }

    /* ---- config ---- */
    int width  = 0;
    int height = 0;

    static constexpr int NUM_TILES_X = 32;
    static constexpr int NUM_TILES_Y = 32;

    /* ---- Gaussian state ---- */
    Gaussian3DParams gaussianParams;

    /* ---- pipeline ---- */
    Pipeline pipeline;

    /* ---- layers ---- */
    GaussActivLayer atvLayer;
    NDCProjectLayer ndcLayer;
    RasterizeLayer  rasLayer;
    MSELossLayer    mseLayer;

    /* ---- optimizer ---- */
    AdamConfig adamConfig;
    int        iterCount = 0;

    /* ---- target image ---- */
    float *d_target_pixels = nullptr;
};
