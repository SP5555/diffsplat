#pragma once
#include <string>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../input/input.h"

/**
 * @brief Base class for all apps.
 * 
 * Handles GLFW/GLAD init, window creation, CUDA device info, the main loop,
 * and display (fullscreen quad + PBO/host-copy paths).
 * 
 * Subclasses implement onStart(), onRender(),
 * and optionally onInput() and onWindowResize().
 * 
 * Subclasses should call displayFrame(pixels) at the end of onRender()
 * to push pixels to screen.
 */
class AppBase
{
public:
    virtual ~AppBase();

    void start();

protected:
    // derived class must call this as: AppBase(w, h, title, resizable)
    AppBase(int width, int height, const std::string &title, bool resizable);

    // override in subclass
    virtual void onStart()  = 0;   // called once before the loop
    virtual void onRender() = 0;   // called every frame

    // override when subclass needs access to input states 
    virtual void onInput() {}
    // override when subclass needs sizes of resized framebuffer 
    virtual void onWindowResize(int width, int height) {};

    // call at the end of onRender() with your CUDA device pixel buffer [H*W*3]
    void displayFrame(const float *d_pixels);

    // window / input state
    GLFWwindow *window = nullptr;
    Input input;
    int width  = 0;
    int height = 0;
    bool resizable = false;

private:
    void initGL();
    void initPBO();
    bool checkCudaGLInterop();

    // used to update PBO/texture on window resize
    void onResize(int newWidth, int newHeight);

    static void glfwErrorCallback(int error, const char *description);

    /* ---- GL objects ---- */
    GLuint vao            = 0;
    GLuint vbo            = 0;
    GLuint texture        = 0;
    GLuint shader_program = 0;
    GLuint pbo            = 0;

    /* ---- CUDA/GL interop ---- */
    cudaGraphicsResource *d_pbo_resource     = nullptr;
    bool                  cudaGLInteropSupported = false;
    std::vector<float>    h_pixels;

    /* ---- loop state ---- */
    double lastFrameTime    = 0.0;
    double timeSinceUpdate  = 0.0;
    int    frameSinceUpdate = 0;
    float  avgFPS           = 0.f;
    std::string title;
};
