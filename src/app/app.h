#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "../input/input.h"
#include "../renderer/compute_renderer.h"

class App
{
public:
    App(int w, int h, const std::string &imagePath);
    ~App();

    void start();

private:
    void handleInput();

    GLFWwindow* window;
    int width = 1280;
    int height = 720;
    std::string imagePath;

    Input input;
    ComputeRenderer renderer;

    double lastFrameTime = 0.0;
    float overlayAccum = 0.f;
    int overlayFrameCount = 0;
    int iterCount = 0;
};