#include "image_fitter.h"

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "../cuda/cuda_check.h"
#include "../io/image_loader.h"
#include "../io/ply_saver.h"
#include "../optimizers/adam.cuh"
#include "../utils/logs.h"
#include "../utils/splat_utils.h"

/* ===== ===== Lifecycle ===== ===== */

void ImageFitter::init(int w, int h)
{
    width  = w;
    height = h;

    log_info("ImageFitter",
        "WindowSize=" + std::to_string(w) + "x" + std::to_string(h) +
        " Tiles=" + std::to_string(NUM_TILES_X) + "x" + std::to_string(NUM_TILES_Y) +
        " MaxPairs=" + std::to_string(MAX_PAIRS)
    );
}

void ImageFitter::loadTargetImage(const std::string &imagePath, int w, int h, int padding)
{
    auto image = ImageLoader::load(imagePath, w, h, padding);
    if (image.pixels.empty())
        log_fatal("ImageFitter", "Failed to load target image: " + imagePath);

    d_target_pixels.allocate(w * h * 3);
    CUDA_CHECK(cudaMemcpy(d_target_pixels, image.pixels.data(),
                          w * h * 3 * sizeof(float), cudaMemcpyHostToDevice));
}

void ImageFitter::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    auto splats = SplatUtils::randomInit(count, width, height, seed);
    gaussian_params.upload(splats);
}

float *ImageFitter::getOutput()
{
    return ras_layer.getOutput();
}

/* ===== ===== Init ===== ===== */

void ImageFitter::initLayers()
{
    int count = gaussian_params.count;

    // allocate forward buffers
    atv_layer.allocate(count);
    ndc_layer.allocate(width, height, count);
    ras_layer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, MAX_PAIRS, count);
    mse_layer.allocate(width, height);

    // allocate grad buffers
    atv_layer.allocateGrad(count);
    ndc_layer.allocateGrad(count);
    ras_layer.allocateGrad(count);

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

void ImageFitter::step()
{
    pipeline.zero_grad();
    pipeline.forward();

    if (!is_optimization_running) return;

    pipeline.backward();
    optimizer.step(gaussian_params, atv_layer.getGradInput());
}

void ImageFitter::savePLY(const std::string &path)
{
    auto splats = gaussian_params.download();
    PLYSaver::save(path, splats, gaussian_params.sh_num_bands);
}
