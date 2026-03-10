#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdint>
#include <vector>
#include <string>

#include "../gaussian/gaussian.h"
#include "../layers/ndc_project_layer.h"
#include "../layers/rasterize_layer.h"
#include "../layers/mse_loss_layer.h"
#include "../optimizers/adam.cuh"

#define NUM_TILES_X 64
#define NUM_TILES_Y 64
#define MAX_PAIRS   (NUM_TILES_X * NUM_TILES_Y * 1024)

class GaussImgFitter
{
public:
    GaussImgFitter() = default;
    ~GaussImgFitter();

    void init(int width, int height);
    void loadTargetImage(const std::string &imagePath, int width, int height, int padding = 0);
    void randomInitGaussians(int count, int seed = -1);
    void render();
    void freeCUDA();
    void freeGL();
    int  getIterCount();

private:
    void initGL();
    void initCUDA();
    void initPBO();
    void initLayers();
    void displayFrame();
    bool checkCudaGLInterop();

    int maxPairs() const { return MAX_PAIRS; }

    int width  = 0;
    int height = 0;

    // --- Gaussian data ---
    GaussianParams      gaussianParams;
    GaussianOptState    gaussianOptState;

    // --- Layers ---
    NDCProjectLayer     orthoLayer;
    RasterizeLayer      rasterizeLayer;
    MSELossLayer        lossLayer;

    // --- Adam optimizer ---
    AdamConfig  adamConfig;
    uint32_t    iterCount = 0;

    // --- Target image (owned here, wired into the loss layer) ---
    float *d_target_pixels = nullptr;

    // --- Display ---
    std::vector<float> h_pixels; // fallback host buffer (cross-device)
    bool cudaGLInteropSupported = false;

    // --- OpenGL objects ---
    GLuint texture        = 0;
    GLuint vao            = 0;
    GLuint vbo            = 0;
    GLuint shader_program = 0;

    // --- PBO (CUDA-GL interop) ---
    GLuint                 pbo            = 0;
    cudaGraphicsResource_t d_pbo_resource = nullptr;
};