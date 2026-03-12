#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

Camera::Camera(float aspect, float fovDegrees, float nearPlane, float farPlane)
    : aspect(aspect)
    , nearP(nearPlane)
    , farP(farPlane)
    , fov(glm::radians(fovDegrees))
{
    updateViewSpaceVectors();
    updateMatrices();
}

/* ===== ===== Update ===== ===== */

bool Camera::update(glm::vec2 mouseDelta, float scrollDelta, bool shiftHeld, float dt)
{
    // clamp large deltas from click jumps
    mouseDelta.x = glm::clamp(mouseDelta.x, -50.f, 50.f);
    mouseDelta.y = glm::clamp(mouseDelta.y, -50.f, 50.f);

    glm::vec3 offset   = position - target;
    glm::vec3 viewDir  = glm::normalize(-offset);
    float     distance = glm::length(offset);

    bool isDirty = false;

    // ===== orbit =====
    if (!shiftHeld && (mouseDelta.x != 0.f || mouseDelta.y != 0.f))
    {
        glm::vec3 dir = glm::normalize(offset);

        float pitch = glm::asin(glm::clamp(dir.y, -1.f, 1.f));
        float yaw   = glm::atan(dir.x, dir.z);

        pitch += mouseDelta.y * rotateSpeed * dt;
        yaw   -= mouseDelta.x * rotateSpeed * dt;

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
    if (shiftHeld && (mouseDelta.x != 0.f || mouseDelta.y != 0.f))
    {
        glm::vec3 translation =
            right * (-mouseDelta.x * distance * panSpeed) +
            up    * ( mouseDelta.y * distance * panSpeed);

        position += translation;
        target   += translation;
        isDirty   = true;
    }

    // ===== zoom =====
    if (scrollDelta != 0.f)
    {
        glm::vec3 translation = viewDir * (-scrollDelta * distance * zoomSpeed);
        position += translation;

        float newDist = glm::distance(position, target);
        if (newDist < minDistance)
            position = target + (-viewDir) * minDistance;
        if (newDist > maxDistance)
            position = target + (-viewDir) * maxDistance;

        isDirty = true;
    }

    if (!isDirty) return false;
    updateViewSpaceVectors();
    updateMatrices();
    return true;
}

/* ===== ===== Resize ===== ===== */

void Camera::setAspect(float newAspect)
{
    aspect  = newAspect;
    pMatrix = glm::perspective(fov, aspect, nearP, farP);
}

/* ===== ===== Helpers ===== ===== */

void Camera::updateViewSpaceVectors()
{
    forward = glm::normalize(target - position);
    right   = glm::normalize(glm::cross(forward, glm::vec3(0.f, 1.f, 0.f)));
    up      = glm::normalize(glm::cross(right, forward));
}

void Camera::updateMatrices()
{
    vMatrix = glm::lookAt(position, target, up);
    pMatrix = glm::perspective(fov, aspect, nearP, farP);
}
