#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#include "gauss_imgfitter.h"
#include "../loaders/image_loader.h"
#include "../cuda/cuda_check.h"
#include "../utils/logs.h"
#include "../utils/splat_utils.h"
#include "../optimizers/adam.cuh"

/* ===== ===== Lifecycle ===== ===== */

void GaussImgFitter::init(int w, int h)
{
    width  = w;
    height = h;

    log_info("GaussImgFitter",
        "WindowSize=" + std::to_string(w) + "x" + std::to_string(h) +
        " Tiles=" + std::to_string(NUM_TILES_X) + "x" + std::to_string(NUM_TILES_Y) +
        " MaxPairs=" + std::to_string(getMaxPairs())
    );
}

void GaussImgFitter::loadTargetImage(const std::string &imagePath, int w, int h, int padding)
{
    auto image = ImageLoader::load(imagePath, w, h, padding);
    if (image.pixels.empty())
        log_fatal("GaussImgFitter", "Failed to load target image: " + imagePath);

    d_target_pixels.allocate(w * h * 3);
    cudaMemcpy(d_target_pixels, image.pixels.data(),
               w * h * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void GaussImgFitter::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    auto splats = SplatUtils::randomInit(count, width, height, seed);
    gaussian_params.upload(splats);
}

float *GaussImgFitter::getOutput()
{
    return ras_layer.getOutput();
}

/* ===== ===== Init ===== ===== */

void GaussImgFitter::initLayers()
{
    int count = gaussian_params.count;

    // allocate
    atv_layer.allocate(count);
    ndc_layer.allocate(width, height, count);
    ras_layer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, getMaxPairs(), count);
    mse_layer.allocate(width, height);

    // wire forward
    atv_layer.setInput(&gaussian_params);
    ndc_layer.setInput(&atv_layer.getOutput());
    ras_layer.setInput(&ndc_layer.getOutput());
    mse_layer.setInput(ras_layer.getOutput());
    mse_layer.setTarget(d_target_pixels);

    // wire backward
    ras_layer.setGradOutput(mse_layer.getGradInput());
    ndc_layer.setGradOutput(&ras_layer.getGradInput());
    atv_layer.setGradOutput(&ndc_layer.getGradInput());

    // optimizer state
    optimizer.init(count);

    // register in pipeline
    pipeline.add(&atv_layer);
    pipeline.add(&ndc_layer);
    pipeline.add(&ras_layer);
    pipeline.add(&mse_layer);
}

/* ===== ===== Render ===== ===== */

void GaussImgFitter::step()
{
    pipeline.zero_grad();
    pipeline.forward();

    if (!is_optimization_running) return;

    pipeline.backward();
    optimizer.step(gaussian_params, atv_layer.getGradInput());
}
