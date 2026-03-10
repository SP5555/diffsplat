#include "gauss_img_fitter.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <chrono>

#include "../loaders/image_loader.h"
#include "../utils/cuda_utils.cuh"
#include "../optimizers/adam.cuh"

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

/* ===== ===== Lifecycle ===== ===== */

ComputeRenderer::~ComputeRenderer()
{
    freeCUDA();
    freeGL();
}

void ComputeRenderer::init(int w, int h)
{
    width  = w;
    height = h;

    initGL();
    initCUDA();

    cudaGLInteropSupported = checkCudaGLInterop();
    if (cudaGLInteropSupported) {
        initPBO();
    } else {
        // fallback host buffer for cross-device path
        h_pixels.resize(width * height * 3);
    }

    std::cout << "[ComputeRenderer] Init " << w << "x" << h
              << " tiles=" << NUM_TILES_X << "x" << NUM_TILES_Y
              << " maxPairs=" << maxPairs()
              << " display=" << (cudaGLInteropSupported ? "PBO" : "host-copy") << "\n";
}

void ComputeRenderer::loadTargetImage(const std::string &imagePath, int w, int h, int padding)
{
    auto image = ImageLoader::load(imagePath, w, h, padding);
    if (image.pixels.empty())
        throw std::runtime_error("Failed to load target image: " + imagePath);

    cudaMalloc(&d_target_pixels, w * h * 3 * sizeof(float));
    cudaMemcpy(d_target_pixels, image.pixels.data(),
               w * h * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void ComputeRenderer::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    gaussianParams = GaussianParams::randomInit(count, width, height, seed);
    gaussianOptState.allocateDeviceMem(gaussianParams.count);

    // layers can only be wired after gaussians are initialized
    initLayers();
}

int ComputeRenderer::getIterCount()
{
    return iterCount;
}

/* ===== ===== Init ===== ===== */

void ComputeRenderer::initGL()
{
    // fullscreen quad
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    shader_program = buildDisplayProgram();

    // output texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void ComputeRenderer::initCUDA()
{
    // cleaned this up too much, now it's chilling at the beach. 
    return;
}

void ComputeRenderer::initPBO()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(float), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&d_pbo_resource, pbo,
                                                    cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "[ComputeRenderer] PBO registration failed, falling back to host copy: "
                  << cudaGetErrorString(err) << "\n";
        glDeleteBuffers(1, &pbo);
        pbo = 0;
        cudaGLInteropSupported = false;
    }
}

void ComputeRenderer::initLayers()
{
    int count = gaussianParams.count;

    // allocate
    orthoLayer.allocate(count);
    rasterizeLayer.allocate(width, height, NUM_TILES_X, NUM_TILES_Y, maxPairs(), count);
    lossLayer.allocate(width, height);

    // wire forward path
    // input -> layer -> output 
    orthoLayer.setInput(&gaussianParams);
    rasterizeLayer.setInput(&orthoLayer.getOutput());
    lossLayer.setInput(rasterizeLayer.getOutput());
    lossLayer.setTarget(d_target_pixels);

    // wire backward path
    // grad input <- layer <- grad output
    // I know, vocabulary isn't the best. That's why this comment exists.
    rasterizeLayer.setGradOutput(lossLayer.getGradInput());
    orthoLayer.setGradOutput(&rasterizeLayer.getGradInput());
    orthoLayer.setGradInput(&gaussianOptState);
}

bool ComputeRenderer::checkCudaGLInterop()
{
    unsigned int deviceCount = 0;
    int glDevices[4];
    cudaGLGetDevices(&deviceCount, glDevices, 4, cudaGLDeviceListAll);

    int cudaDevice;
    cudaGetDevice(&cudaDevice);

    for (unsigned int i = 0; i < deviceCount; ++i)
        if (glDevices[i] == cudaDevice)
            return true;

    return false;
}

/* ===== ===== Render ===== ===== */

void ComputeRenderer::render()
{
    // zero all gradients before each frame
    lossLayer.zero_grad();
    rasterizeLayer.zero_grad();
    orthoLayer.zero_grad();
    gaussianOptState.zero_grad();

    // forward
    orthoLayer.forward(width, height);
    rasterizeLayer.forward(width, height);

    // backward
    lossLayer.backward();
    rasterizeLayer.backward(width, height);
    orthoLayer.backward(width, height);

    // optimizer step
    launchAdam(gaussianParams, gaussianOptState, adamConfig, ++iterCount);

    displayFrame();
}

void ComputeRenderer::displayFrame()
{
    const float *pixels = rasterizeLayer.getOutput();

    if (cudaGLInteropSupported)
    {
        // D2D copy into PBO
        cudaGraphicsMapResources(1, &d_pbo_resource);
        float *d_pbo  = nullptr;
        size_t pbo_size = 0;
        cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &pbo_size, d_pbo_resource);
        cudaMemcpy(d_pbo, pixels, width * height * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &d_pbo_resource);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        // fallback: GPU -> CPU -> GPU
        // CUDA Mem to PBO is not possible if GL and CUDA are on different devices
        // e.g. AMD GL + NVIDIA CUDA
        cudaMemcpy(h_pixels.data(), pixels,
                   width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, h_pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

/* ===== ===== Cleanup ===== ===== */

void ComputeRenderer::freeCUDA()
{
    gaussianParams.free();
    gaussianOptState.free();

    orthoLayer.free();
    rasterizeLayer.free();
    lossLayer.free();

    CUDA_FREE(d_target_pixels);
}

void ComputeRenderer::freeGL()
{
    if (pbo)
    {
        if (d_pbo_resource)
        {
            cudaGraphicsUnregisterResource(d_pbo_resource);
            d_pbo_resource = nullptr;
        }
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }

    if (texture)        { glDeleteTextures(1, &texture);   texture = 0; }
    if (vao)            { glDeleteVertexArrays(1, &vao);   vao = 0; }
    if (vbo)            { glDeleteBuffers(1, &vbo);        vbo = 0; }
    if (shader_program) { glDeleteProgram(shader_program); shader_program = 0; }
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
