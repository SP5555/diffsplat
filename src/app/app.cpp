#include <iostream>
#include <stdexcept>

#include "app.h"
#include "../gaussian/gaussian.h"
#include "../renderer/compute_renderer.h"

static void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "[GLFW] Error " << error << ": " << description << std::endl;
}

App::App(int w, int h, const std::string &imagePath) :
    width(w),
    height(h),
    imagePath(imagePath)
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(width, height, "Diffsplat", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties_v2(&deviceProp, 0);

    std::cout << "[App] OpenGL " << glGetString(GL_VERSION) << "\n"
              << "[App] OpenGL Renderer: " << glGetString(GL_RENDERER) << "\n"
              << "[App] CUDA Device: " << deviceProp.name << "\n";

    Input::install(window, &input);
    renderer.loadTargetImage(imagePath, width, height);
    renderer.init(width, height);
}

App::~App()
{
    if (window)
        glfwDestroyWindow(window);

    glfwTerminate();

    renderer.free();
}

void App::start()
{
    lastFrameTime = glfwGetTime();

    renderer.randomInitGaussians(80000);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        handleInput();
        renderer.render();

        glfwSwapBuffers(window);
        input.flush();
    }
}

void App::handleInput()
{
    // muted for now
    // if (input.mouseLeft)
    // {
    //     std::cout << "[App] Mouse Left Held at (" << input.mousePos.x << ", " << input.mousePos.y << ")\n";
    // }
    // if (input.mouseRight)
    // {
    //     std::cout << "[App] Mouse Right Held at (" << input.mousePos.x << ", " << input.mousePos.y << ")\n";
    // }
    // if (input.scrollDelta != 0.f)
    // {
    //     std::cout << "[App] Scroll Delta: " << input.scrollDelta << "\n";
    // }
}