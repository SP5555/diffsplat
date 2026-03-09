#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "../gaussian/gaussian.h"
#include "../kernels/adam.cuh"

// screen is divided into this amount of tiles
// each tile gets a list of contributing splats
// which are sorted by depth and rendered in the forward pass
#define NUM_TILES_X 64
#define NUM_TILES_Y 64
#define MAX_PAIRS (NUM_TILES_X * NUM_TILES_Y * 512) // estimate

class ComputeRenderer
{
public:
    ComputeRenderer() = default;
    ~ComputeRenderer();

    void init(int width, int height);
    void loadTargetImage(const std::string &imagePath, int width, int height);
    void randomInitGaussians(int count, int seed = -1);
    void render();
    void free();

private:
    void initGL();
    void initCUDA();
    void uploadToTexture();

    int width = 0;
    int height = 0;

    int maxPairs() const
    {
        return MAX_PAIRS;
    }

    // Gaussian data
    GaussianParams gaussianParams;
    GaussianOptState gaussianOptState;
    
    // CUDA buffers
    float*      d_pixels        = nullptr;  // [H * W * 3]
    float*      d_T_final       = nullptr;  // [H * W]
    int*        d_n_contrib     = nullptr;  // [H * W]
    
    uint64_t*   d_keys          = nullptr;  // [max_pairs] unsorted
    uint32_t*   d_values        = nullptr;  // [max_pairs] unsorted
    uint64_t*   d_keys_sorted   = nullptr;  // [max_pairs]
    uint32_t*   d_values_sorted = nullptr;  // [max_pairs]
    uint32_t*   d_pair_count    = nullptr;  // [1] atomic
    int2*       d_tile_ranges   = nullptr;  // [NUM_TILES_X * NUM_TILES_Y]

    float*      d_target_pixels = nullptr;  // [H * W * 3]
    
    // temporary stuff interally used by CUB Radix Sort
    // for some reason, we have to provide them
    void*       d_sort_temp     = nullptr;
    size_t      sort_temp_bytes = 0;
    
    // Adam optimizer config
    AdamConfig adamConfig; // default
    uint32_t iterCount = 0; // for bias correction in Adam

    // OpenGL objects
    GLuint texture          = 0;
    GLuint vao              = 0;
    GLuint vbo              = 0;
    GLuint shader_program   = 0;
};
