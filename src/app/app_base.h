#pragma once
#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "../input/input.h"

/**
 * @brief Base class for all apps.
 * 
 * Handles GLFW/GLAD initialization, window creation, CUDA device info,
 * and the main loop skeleton. Subclasses implement onStart() and onRender().
 */
class AppBase
{
public:
    AppBase(int width, int height, const std::string &title);
    virtual ~AppBase();

    void start();

protected:
    // override in subclass
    virtual void onStart()  = 0;   // called once before the loop
    virtual void onRender() = 0;   // called every frame
    virtual void onInput()  {}     // called every frame after polling

    // window state
    GLFWwindow *window = nullptr;
    Input input;
    int width  = 0;
    int height = 0;

private:
    static void glfwErrorCallback(int error, const char *description);

    double lastFrameTime    = 0.0;
    double timeSinceUpdate  = 0.0;
    int    frameSinceUpdate = 0;
    float  avgFPS           = 0.f;
    std::string title;
};
