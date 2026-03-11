#include "app_base.h"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

/* ===== ===== Lifecycle ===== ===== */

void AppBase::glfwErrorCallback(int error, const char *description)
{
    std::cerr << "[GLFW] Error " << error << ": " << description << "\n";
}

AppBase::AppBase(int width, int height, const std::string &title)
    : width(width), height(height), title(title)
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties_v2(&deviceProp, 0);

    std::cout << "[App] OpenGL " << glGetString(GL_VERSION) << "\n"
              << "[App] Renderer: " << glGetString(GL_RENDERER) << "\n"
              << "[App] CUDA Device: " << deviceProp.name << "\n";

    Input::install(window, &input);
}

AppBase::~AppBase()
{
    if (window)
        glfwDestroyWindow(window);
    glfwTerminate();
}

/* ===== ===== Main Loop ===== ===== */

void AppBase::start()
{
    onStart();
    lastFrameTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        double currentTime = glfwGetTime();
        double deltaTime   = currentTime - lastFrameTime;
        lastFrameTime      = currentTime;
        timeSinceUpdate   += deltaTime;
        frameSinceUpdate++;

        // update window title with FPS every 100ms
        if (timeSinceUpdate >= 0.1)
        {
            avgFPS = avgFPS * 0.4f + (frameSinceUpdate / (float)timeSinceUpdate) * 0.6f;
            char buf[128];
            snprintf(buf, sizeof(buf), "%s [FPS: %.1f]", title.c_str(), avgFPS);
            glfwSetWindowTitle(window, buf);
            timeSinceUpdate  = 0.0;
            frameSinceUpdate = 0;
        }

        onInput();
        onRender();

        glfwSwapBuffers(window);
        input.flush();
    }
}
