#include "gauss_img_fitter.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#include "../loaders/image_loader.h"
#include "../utils/cuda_utils.cuh"
#include "../optimizers/adam.cuh"

/* ===== ===== Lifecycle ===== ===== */

void GaussImgFitter::init(int w, int h)
{
    width  = w;
    height = h;

    std::cout << "[GaussImgFitter] Init " << w << "x" << h
              << " tiles=" << NUM_TILES_X << "x" << NUM_TILES_Y
              << " maxPairs=" << maxPairs() << "\n";
}

void GaussImgFitter::loadTargetImage(const std::string &imagePath, int w, int h, int padding)
{
    auto image = ImageLoader::load(imagePath, w, h, padding);
    if (image.pixels.empty())
        throw std::runtime_error("Failed to load target image: " + imagePath);

    d_target_pixels.allocate(w * h * 3);
    cudaMemcpy(d_target_pixels, image.pixels.data(),
               w * h * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void GaussImgFitter::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    gaussianParams = Gaussian3DParams::randomInit(count, width, height, seed);
}

float *GaussImgFitter::getOutput()
{
    return rasLayer.getOutput();
}

/* ===== ===== Init ===== ===== */

void GaussImgFitter::initLayers()
{
    int count = gaussianParams.count;

    // allocate
    atvLayer.allocate(count);
    ndcLayer.allocate(width, height, count);
    rasLayer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, maxPairs(), count);
    mseLayer.allocate(width, height);

    // wire forward
    atvLayer.setInput(&gaussianParams);
    ndcLayer.setInput(&atvLayer.getOutput());
    rasLayer.setInput(&ndcLayer.getOutput());
    mseLayer.setInput(rasLayer.getOutput());
    mseLayer.setTarget(d_target_pixels);

    // wire backward
    rasLayer.setGradOutput(mseLayer.getGradInput());
    ndcLayer.setGradOutput(&rasLayer.getGradInput());
    atvLayer.setGradOutput(&ndcLayer.getGradInput());

    // register in pipeline
    pipeline.add(&atvLayer);
    pipeline.add(&ndcLayer);
    pipeline.add(&rasLayer);
    pipeline.add(&mseLayer);
}

/* ===== ===== Render ===== ===== */

void GaussImgFitter::render()
{
    pipeline.zero_grad();
    pipeline.forward();
    if (iterCount % 10 == 0)
        printf("Iter %d: Loss = %.8f\n", iterCount, mseLayer.getLoss());
    pipeline.backward();

    // optimizer step
    launchAdam(gaussianParams, atvLayer.getGradInput(), adamConfig, ++iterCount);
}
