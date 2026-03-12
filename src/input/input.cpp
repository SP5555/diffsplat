#include <unordered_map>
#include "input.h"

static std::unordered_map<GLFWwindow*, Input*> s_inputs;

void Input::flush()
{
    mouseDelta = {0.f, 0.f};
    scrollDelta = 0.f;
}

void Input::install(GLFWwindow *window, Input *input)
{
    s_inputs[window] = input;
    glfwSetMouseButtonCallback(window, cbMouseButton);
    glfwSetCursorPosCallback(window, cbMouseMove);
    glfwSetScrollCallback(window, cbMouseScroll);
    glfwSetKeyCallback(window, cbKey);
}

void Input::cbMouseButton(GLFWwindow *window, int button, int action, int mods)
{
    Input *input = s_inputs[window];
    bool pressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        input->mouseLeftPressed = pressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        input->mouseRightPressed = pressed;
    }
}

void Input::cbMouseMove(GLFWwindow *window, double xpos, double ypos)
{
    Input *input = s_inputs[window];
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
    Input *input = s_inputs[window];
    input->scrollDelta += static_cast<float>(yoffset);
}

void Input::cbKey(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Input *input = s_inputs[window];
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
    {
        input->shiftPressed = (action != GLFW_RELEASE);
    }
}