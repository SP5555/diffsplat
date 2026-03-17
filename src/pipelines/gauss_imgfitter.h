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
    ~GaussImgFitter() {};

    void init(int width, int height);
    void loadTargetImage(const std::string &imagePath, int w, int h, int padding = 0);
    void randomInitGaussians(int count, int seed = -1);
    void initLayers();

    void step();

    float *getOutput();
    int    getIterCount() const { return optimizer.getStepCount(); }
    float  getLoss() const      { return mse_layer.getLoss(); }
private:
    int getMaxPairs() const { return (1 << 20); }

    /* ---- config ---- */
    int width  = 0;
    int height = 0;

    static constexpr int NUM_TILES_X = 32;
    static constexpr int NUM_TILES_Y = 32;

    /* ---- Gaussian state ---- */
    Gaussian3DParams gaussian_params;

    /* ---- pipeline ---- */
    Pipeline pipeline;

    /* ---- layers ---- */
    GaussActivLayer atv_layer;
    NDCProjectLayer ndc_layer;
    RasterizeLayer  ras_layer;
    MSELossLayer    mse_layer;

    /* ---- optimizer ---- */
    Adam optimizer;

    /* ---- target image ---- */
    CudaBuffer<float> d_target_pixels;
};
