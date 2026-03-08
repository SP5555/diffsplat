#include "input.h"

void Input::flush()
{
    mouseDelta = {0.f, 0.f};
    scrollDelta = 0.f;
}

void Input::install(GLFWwindow *window, Input *input)
{
    glfwSetWindowUserPointer(window, input);
    glfwSetMouseButtonCallback(window, cbMouseButton);
    glfwSetCursorPosCallback(window, cbMouseMove);
    glfwSetScrollCallback(window, cbMouseScroll);
}

void Input::cbMouseButton(GLFWwindow *window, int button, int action, int mods)
{
    Input *input = static_cast<Input *>(glfwGetWindowUserPointer(window));
    bool pressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        input->mouseLeft = pressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        input->mouseRight = pressed;
    }
}

void Input::cbMouseMove(GLFWwindow *window, double xpos, double ypos)
{
    Input *input = static_cast<Input *>(glfwGetWindowUserPointer(window));
    glm::vec2 newPos = {static_cast<float>(xpos), static_cast<float>(ypos)};

    if (input->firstMouse)
    {
        input->lastMousePos = newPos;
        input->firstMouse = false;
    }

    input->mouseDelta += (newPos - input->lastMousePos);
    input->lastMousePos = newPos;
    input->mousePos = newPos;
}

void Input::cbMouseScroll(GLFWwindow *window, double xoffset, double yoffset)
{
    Input *input = static_cast<Input *>(glfwGetWindowUserPointer(window));
    input->scrollDelta += static_cast<float>(yoffset);
}