#include <iostream>
#include "fly_camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

FlyCamera::FlyCamera(float aspect, float fovDegrees, float nearPlane, float farPlane)
    : aspect(aspect)
    , nearP(nearPlane)
    , farP(farPlane)
    , fov(glm::radians(fovDegrees))
{
    std::cout << "[FlyCamera] Controls:\n"
              << "\e[1;36m"
              << "  W/A/S/D           -> Move/Strafe\n"
              << "  R/F               -> Move up/down\n"
              << "  Q/E               -> Roll left/right\n"
              << "  Left Click + Drag -> Look around (yaw/pitch)\n"
              << "  Shift             -> Fast move\n"
              << "  Ctrl              -> Slow move\n"
              << "\e[0m";

    updateViewSpaceVectors();
    updateMatrices();
}

/* ===== ===== Update ===== ===== */

bool FlyCamera::update(const Input &input, float dt)
{
    // clamp large deltas from click jumps
    float mouseDelta_x =  glm::clamp(input.mouseDelta.x, -50.f, 50.f);
    float mouseDelta_y = -glm::clamp(input.mouseDelta.y, -50.f, 50.f);

    float speedMult = 1.f;
    if (input.isShiftDown())
        speedMult *= speedMultShift;
    if (input.isCtrlDown())
        speedMult *= speedMultCtrl;
    
    float moveDelta = moveSpeed * speedMult * dt;
    float rollDelta = rollSpeed * speedMult * dt;
    float lookDelta = lookSpeed * speedMult * dt;

    bool isDirty = false;

    // WASD forward/strafe on the local plane
    if (input.isKeyDown(GLFW_KEY_W)) { position += planeForward * moveDelta; isDirty = true; }
    if (input.isKeyDown(GLFW_KEY_S)) { position -= planeForward * moveDelta; isDirty = true; }
    if (input.isKeyDown(GLFW_KEY_A)) { position -= planeRight   * moveDelta; isDirty = true; }
    if (input.isKeyDown(GLFW_KEY_D)) { position += planeRight   * moveDelta; isDirty = true; }

    // RF up/down
    if (input.isKeyDown(GLFW_KEY_R)) { position -= planeUp      * moveDelta; isDirty = true; }
    if (input.isKeyDown(GLFW_KEY_F)) { position += planeUp      * moveDelta; isDirty = true; }

    // QE roll
    if (input.isKeyDown(GLFW_KEY_Q)) {
        orientation = orientation * glm::angleAxis( rollDelta, glm::vec3(0.f, 0.f, -1.f));
        isDirty = true;
    }
    if (input.isKeyDown(GLFW_KEY_E)) {
        orientation = orientation * glm::angleAxis(-rollDelta, glm::vec3(0.f, 0.f, -1.f));
        isDirty = true;
    }

    // mouse look (yaw/pitch)
    if (input.mouseLeftPressed)
    {
        if (mouseDelta_x != 0.f) {
            orientation = orientation * glm::angleAxis(
                -mouseDelta_x * lookDelta,
                glm::vec3(0.f, 1.f, 0.f) // local up
            );
            isDirty = true;
        }
        if (mouseDelta_y != 0.f) {
            pitch += -mouseDelta_y * lookDelta;
            pitch = glm::clamp(pitch, MIN_PITCH, MAX_PITCH);
            isDirty = true;
        }
    }

    if (!isDirty) return false;
    updateViewSpaceVectors();
    updateMatrices();
    return true;
}

/* ===== ===== Resize ===== ===== */

void FlyCamera::setAspect(float newAspect)
{
    aspect  = newAspect;
    pMatrix = glm::perspective(fov, aspect, nearP, farP);
}

/* ===== ===== Helpers ===== ===== */

void FlyCamera::updateViewSpaceVectors()
{
    orientation = glm::normalize(orientation);

    planeForward = orientation * glm::vec3( 0.f,  0.f, -1.f);
    planeRight   = orientation * glm::vec3( 1.f,  0.f,  0.f);
    planeUp      = orientation * glm::vec3( 0.f,  1.f,  0.f);
}

void FlyCamera::updateMatrices()
{
    glm::vec3 viewDir = glm::normalize(
        planeForward * glm::cos(pitch) + planeUp * glm::sin(pitch)
    );

    glm::vec3 viewUp = glm::normalize(glm::cross(planeRight, viewDir));

    vMatrix = glm::lookAt(position, position + viewDir, viewUp);
    pMatrix = glm::perspective(fov, aspect, nearP, farP);
}