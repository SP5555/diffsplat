#include "compute_renderer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "../loaders/image_loader.h"
#include "../utils/cuda_utils.h"
#include "../kernels/tile_assign.cuh"
#include "../kernels/sort.cuh"
#include "../kernels/forward.cuh"
#include "../kernels/backward.cuh"
#include "../kernels/adam.cuh"

static const float QUAD[] = {
    // x,    y,    u,    v,
    -1.f, -1.f,  0.f,  1.f,
     1.f, -1.f,  1.f,  1.f,
     1.f,  1.f,  1.f,  0.f,
    -1.f, -1.f,  0.f,  1.f,
     1.f,  1.f,  1.f,  0.f,
    -1.f,  1.f,  0.f,  0.f,
};

static GLuint compileShader(GLenum type, const char *src);
static GLuint buildDisplayProgram();

ComputeRenderer::~ComputeRenderer()
{
    freeCUDA();
    freeGL();
}

void ComputeRenderer::init(int w, int h)
{
    width = w;
    height = h;
    initGL();
    initCUDA();
    std::cout << "[ComputeRenderer] Init " << w << "x" << h
              << " tiles=" << NUM_TILES_X << "x" << NUM_TILES_Y
              << " maxPairs=" << maxPairs() << "\n";
}

// Super Ultra Boilerplate Pro-Max OpenGL DOOM
void ComputeRenderer::initGL()
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    shader_program = buildDisplayProgram();

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,
                 width, height, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void ComputeRenderer::initCUDA()
{
    int pixels = width * height;
    int pairs = maxPairs();
    int numTiles = NUM_TILES_X * NUM_TILES_Y;

    h_pixels.resize(pixels * 3);

    cudaMalloc(&d_pixels, pixels * 3 * sizeof(float));
    cudaMalloc(&d_T_final, pixels * sizeof(float));
    cudaMalloc(&d_n_contrib, pixels * sizeof(int));

    cudaMalloc(&d_keys, pairs * sizeof(uint64_t));
    cudaMalloc(&d_values, pairs * sizeof(uint32_t));
    cudaMalloc(&d_pair_count, sizeof(uint32_t));

    cudaMalloc(&d_keys_sorted, pairs * sizeof(uint64_t));
    cudaMalloc(&d_values_sorted, pairs * sizeof(uint32_t));

    cudaMalloc(&d_tile_ranges, numTiles * sizeof(int2));
}

void ComputeRenderer::loadTargetImage(const std::string &imagePath, int width, int height)
{
    auto image = ImageLoader::load(imagePath, width, height);
    if (image.pixels.empty())
    {
        throw std::runtime_error("Failed to load target image: " + imagePath);
    }

    cudaMalloc(&d_target_pixels, width * height * 3 * sizeof(float));
    cudaMemcpy(
        d_target_pixels, image.pixels.data(),
        width * height * 3 * sizeof(float),
        cudaMemcpyHostToDevice
    );
}

void ComputeRenderer::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    gaussianParams = GaussianParams::randomInit(count, width, height, seed);
    gaussianOptState.allocateDeviceMem(gaussianParams.count);
}

// Render
void ComputeRenderer::render()
{
    int numTiles = NUM_TILES_X * NUM_TILES_Y;

    // clear per-frame buffers
    cudaMemset(d_pixels, 0, width * height * 3 * sizeof(float));
    cudaMemset(d_pair_count, 0, sizeof(uint32_t));
    cudaMemset(d_tile_ranges, 0, numTiles * sizeof(int2));

    gaussianOptState.zeroGradients();

    // tile assignment: emit (key, value) pairs
    launchTileAssign(
        gaussianParams,
        d_keys,
        d_values,
        d_pair_count,
        maxPairs(),
        NUM_TILES_X,
        NUM_TILES_Y,
        width,
        height
    );

    // read back pair count (need it on CPU for sort)
    uint32_t pair_count = 0;
    cudaMemcpy(&pair_count, d_pair_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (pair_count == 0)
    {
        // no splats visible, just clear to black and display
        displayFrame();
    }
    else
    {
        // sort pairs by key (tile_id | depth)
        launchSort(
            d_keys,
            d_values,
            d_keys_sorted,
            d_values_sorted,
            pair_count,
            &d_sort_temp,
            &sort_temp_bytes
        );

        // build tile ranges from sorted keys
        launchBuildTileRanges(
            d_keys_sorted,
            d_tile_ranges,
            pair_count,
            numTiles
        );

        // forward rasterizer
        launchForward(
            gaussianParams,
            d_values_sorted,
            d_tile_ranges,
            d_pixels,
            d_T_final,
            d_n_contrib,
            NUM_TILES_X,
            NUM_TILES_Y,
            width,
            height
        );

        // backward pass to compute gradients
        launchBackward(
            gaussianParams,
            gaussianOptState,
            d_target_pixels,
            d_values_sorted,
            d_tile_ranges,
            d_pixels,
            d_T_final,
            d_n_contrib,
            NUM_TILES_X,
            NUM_TILES_Y,
            width,
            height
        );

        // optimizer step (Adam)
        launchAdam(
            gaussianParams,
            gaussianOptState,
            adamConfig,
            ++iterCount
        );

        displayFrame();
    }
}

void ComputeRenderer::freeCUDA()
{
    gaussianParams.free();
    gaussianOptState.free();

    CUDA_FREE(d_pixels);
    CUDA_FREE(d_T_final);
    CUDA_FREE(d_n_contrib);
    CUDA_FREE(d_keys);
    CUDA_FREE(d_values);
    CUDA_FREE(d_pair_count);
    CUDA_FREE(d_keys_sorted);
    CUDA_FREE(d_values_sorted);
    CUDA_FREE(d_tile_ranges);
    CUDA_FREE(d_sort_temp);
    CUDA_FREE(d_target_pixels);
}

void ComputeRenderer::freeGL()
{
    if (texture)        { glDeleteTextures(1, &texture);    texture = 0; }
    if (vao)            { glDeleteVertexArrays(1, &vao);    vao = 0; }
    if (vbo)            { glDeleteBuffers(1, &vbo);         vbo = 0; }
    if (shader_program) { glDeleteProgram(shader_program);  shader_program = 0; }
}

int ComputeRenderer::getIterCount()
{
    return iterCount;
}

void ComputeRenderer::displayFrame()
{
    cudaMemcpy(
        h_pixels.data(),
        d_pixels,
        width * height * 3 * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    width, height, GL_RGB, GL_FLOAT, h_pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Draw fullscreen quad
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

/* ===== ===== GL Boilerplate ===== ===== */

static const char *VS_SRC = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

static const char *FS_SRC = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = vec4(texture(uTex, vUV).rgb, 1.0);
}
)glsl";

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "[GL] Shader error: " << log << "\n";
    }
    return s;
}

static GLuint buildDisplayProgram()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, VS_SRC);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, FS_SRC);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}