#include <algorithm>
#include <iostream>
#include "arcball_camera.h"
#include <glm/gtc/matrix_transform.hpp>

ArcballCamera::ArcballCamera(float aspect, float fovDegrees, float nearPlane, float farPlane)
    : aspect(aspect)
    , nearP(nearPlane)
    , farP(farPlane)
    , fov(glm::radians(fovDegrees))
{
    std::cout << "[ArcballCamera] Controls:\n" 
              << "\e[1;36m"
              << "  Left Click + Drag         -> Orbit\n"
              << "  Shift + Left Click + Drag -> Pan\n"
              << "  Scroll                    -> Zoom\n"
              << "\e[0m";

    updateViewSpaceVectors();
    updateMatrices();
}

/* ===== ===== Update ===== ===== */

bool ArcballCamera::update(const Input &input, float dt)
{
    // clamp large deltas from click jumps
    float mouseDelta_x =  glm::clamp(input.mouseDelta.x, -50.f, 50.f);
    float mouseDelta_y = -glm::clamp(input.mouseDelta.y, -50.f, 50.f);

    glm::vec3 offset   = position - target;
    glm::vec3 viewDir  = glm::normalize(-offset);
    float     distance = glm::length(offset);

    bool isDirty = false;

    // ===== orbit =====
    if (!input.isShiftDown() && input.mouseLeftPressed && (mouseDelta_x != 0.f || mouseDelta_y != 0.f))
    {
        glm::vec3 dir = glm::normalize(offset);

        float pitch = glm::asin(glm::clamp(dir.y, -1.f, 1.f));
        float yaw   = glm::atan(dir.x, dir.z);

        pitch += mouseDelta_y * rotateSpeed * dt;
        yaw   -= mouseDelta_x * rotateSpeed * dt;

        pitch = glm::clamp(pitch, MIN_PITCH, MAX_PITCH);
        yaw   = glm::mod(yaw, glm::two_pi<float>());

        glm::vec3 newDir = {
            glm::sin(yaw) * glm::cos(pitch),
            glm::sin(pitch),
            glm::cos(yaw) * glm::cos(pitch)
        };

        position = target + newDir * distance;
        isDirty  = true;
    }

    // ===== pan =====
    if (input.isShiftDown() && input.mouseLeftPressed && (mouseDelta_x != 0.f || mouseDelta_y != 0.f))
    {
        glm::vec3 translation =
            right * (-mouseDelta_x * distance * panSpeed) +
            up    * ( mouseDelta_y * distance * panSpeed);

        position += translation;
        target   += translation;
        isDirty   = true;
    }

    // ===== zoom =====
    if (input.scrollDelta != 0.f)
    {
        glm::vec3 translation = viewDir * (input.scrollDelta * distance * zoomSpeed);
        position += translation;

        float newDist = glm::distance(position, target);
        if (newDist < zoomMinDist)
            position = target + (-viewDir) * zoomMinDist;
        if (newDist > zoomMaxDist)
            position = target + (-viewDir) * zoomMaxDist;

        isDirty = true;
    }

    if (!isDirty) return false;
    updateViewSpaceVectors();
    updateMatrices();
    return true;
}

/* ===== ===== Resize ===== ===== */

void ArcballCamera::setAspect(float newAspect)
{
    aspect  = newAspect;
    pMatrix = glm::perspective(fov, aspect, nearP, farP);
}

/* ===== ===== Helpers ===== ===== */

void ArcballCamera::updateViewSpaceVectors()
{
    forward = glm::normalize(target - position);
    right   = glm::normalize(glm::cross(forward, glm::vec3(0.f, 1.f, 0.f)));
    up      = glm::normalize(glm::cross(right, forward));
}

void ArcballCamera::updateMatrices()
{
    vMatrix = glm::lookAt(position, target, up);
    pMatrix = glm::perspective(fov, aspect, nearP, farP);
}
