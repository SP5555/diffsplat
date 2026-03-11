#include "gauss_img_fitter.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#include "../loaders/image_loader.h"
#include "../utils/cuda_utils.cuh"
#include "../optimizers/adam.cuh"

/* ===== ===== Lifecycle ===== ===== */

GaussImgFitter::~GaussImgFitter()
{
    free();
}

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

    cudaMalloc(&d_target_pixels, w * h * 3 * sizeof(float));
    cudaMemcpy(d_target_pixels, image.pixels.data(),
               w * h * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void GaussImgFitter::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    gaussianParams = Gaussian3DParams::randomInit(count, width, height, seed);

    // layers can only be wired after gaussians are initialized
    initLayers();
}

const float *GaussImgFitter::getOutput() const
{
    return rasLayer.getOutput();
}

/* ===== ===== Init ===== ===== */

void GaussImgFitter::initLayers()
{
    int count = gaussianParams.count;

    // allocate
    covLayer.allocate(count);
    ndcLayer.allocate(width, height, count);
    rasLayer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, maxPairs(), count);
    mseLayer.allocate(width, height);

    // wire forward
    covLayer.setInput(&gaussianParams);
    ndcLayer.setInput(&covLayer.getOutput());
    rasLayer.setInput(&ndcLayer.getOutput());
    mseLayer.setInput(rasLayer.getOutput());
    mseLayer.setTarget(d_target_pixels);

    // wire backward
    rasLayer.setGradOutput(mseLayer.getGradInput());
    ndcLayer.setGradOutput(&rasLayer.getGradInput());
    covLayer.setGradOutput(&ndcLayer.getGradInput());
}

/* ===== ===== Render ===== ===== */

void GaussImgFitter::render()
{
    // zero all gradients
    mseLayer.zero_grad();
    rasLayer.zero_grad();
    ndcLayer.zero_grad();
    covLayer.zero_grad();

    // forward
    covLayer.forward();
    ndcLayer.forward();
    rasLayer.forward();
    // only needed for logging:
    // float loss = mseLayer.forward();
    // printf("Iter %d: Loss = %.8f\n", iterCount, loss);

    // backward
    mseLayer.backward();
    rasLayer.backward();
    ndcLayer.backward();
    covLayer.backward();

    // optimizer step
    launchAdam(gaussianParams, covLayer.getGradInput(), adamConfig, ++iterCount);
}

/* ===== ===== Cleanup ===== ===== */

void GaussImgFitter::free()
{
    gaussianParams.free();

    covLayer.free();
    ndcLayer.free();
    rasLayer.free();
    mseLayer.free();

    CUDA_FREE(d_target_pixels);
}
