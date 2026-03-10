#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "../input/input.h"
#include "../pipelines/gauss_img_fitter.h"

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
    GaussImgFitter renderer;

    double lastFrameTime = 0.0;
    int frameSinceUpdate = 0;
    double timeSinceUpdate = 0.0;
};