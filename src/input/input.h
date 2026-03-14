#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_set>

class Input
{
public:
    // mouse
    glm::vec2 mousePos   = {0.f, 0.f};
    glm::vec2 mouseDelta = {0.f, 0.f};
    bool  mouseLeftPressed  = false;
    bool  mouseRightPressed = false;
    float scrollDelta       = 0.f;

    // key query
    bool isKeyDown(int glfwKey) const { return keysDown.count(glfwKey) > 0; }

    bool isShiftDown() const { return isKeyDown(GLFW_KEY_LEFT_SHIFT) || 
                                      isKeyDown(GLFW_KEY_RIGHT_SHIFT); }
    bool isCtrlDown()  const { return isKeyDown(GLFW_KEY_LEFT_CONTROL) || 
                                      isKeyDown(GLFW_KEY_RIGHT_CONTROL); }

    void flush();

    static void install(GLFWwindow *window, Input *input);

private:
    std::unordered_set<int> keysDown;

    glm::vec2 lastMousePos = {0.f, 0.f};
    bool firstMouse = true;

    static void cbMouseButton(GLFWwindow *window, int button, int action, int mods);
    static void cbMouseMove  (GLFWwindow *window, double xpos, double ypos);
    static void cbMouseScroll(GLFWwindow *window, double xoffset, double yoffset);
    static void cbKey        (GLFWwindow *window, int key, int scancode, int action, int mods);
};