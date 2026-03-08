#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

class Input
{
public:
    glm::vec2 mousePos = {0.f, 0.f};
    glm::vec2 mouseDelta = {0.f, 0.f};
    bool mouseLeft = false;
    bool mouseRight = false;
    float scrollDelta = 0.f;

    void flush();

    static void install(GLFWwindow *window, Input *input);

private:
    glm::vec2 lastMousePos = {0.f, 0.f};
    bool firstMouse = true;

    static void cbMouseButton(GLFWwindow *window, int button, int action, int mods);
    static void cbMouseMove(GLFWwindow *window, double xpos, double ypos);
    static void cbMouseScroll(GLFWwindow *window, double xoffset, double yoffset);
};