#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * @brief Orbit camera with pan and zoom.
 * 
 * Controls:
 *   Left drag         -> orbit (yaw / pitch around target)
 *   Shift + left drag -> pan (translate position and target together)
 *   Scroll            -> zoom (dolly along view direction)
 * 
 * Call update() every frame with mouse/scroll deltas.
 * Call getViewMatrix() / getProjectionMatrix() to feed the renderer.
 * Call setAspect() on window resize.
 */
class Camera
{
public:
    Camera(float aspect, float fovDegrees = 60.f,
           float nearPlane = 0.01f, float farPlane = 100.f);

    // call every frame, returns true if matrices changed
    bool update(glm::vec2 mouseDelta, float scrollDelta, bool shiftHeld, float dt);

    // call on window resize
    void setAspect(float aspect);

    const glm::mat4 &getViewMatrix()       const { return vMatrix; }
    const glm::mat4 &getProjectionMatrix() const { return pMatrix; }

    // ---- config ----
    float rotateSpeed = 0.1f;
    float panSpeed    = 0.005f;
    float zoomSpeed   = 0.1f;
    float minDistance = 0.1f;
    float maxDistance = 100.f;

private:
    void updateViewSpaceVectors();
    void updateMatrices();

    // ---- state ----
    glm::vec3 position = {0.f, 0.f, 3.f};
    glm::vec3 target   = {0.f, 0.f, 0.f};

    glm::vec3 forward  = {0.f, 0.f, -1.f};
    glm::vec3 right    = {1.f, 0.f,  0.f};
    glm::vec3 up       = {0.f, 1.f,  0.f};

    // ---- projection params ----
    float fov   = glm::radians(60.f);
    float aspect;
    float nearP;
    float farP;

    // ---- cached matrices ----
    glm::mat4 vMatrix = glm::mat4(1.f);
    glm::mat4 pMatrix = glm::mat4(1.f);

    static constexpr float MIN_PITCH = -glm::half_pi<float>() + 0.01f;
    static constexpr float MAX_PITCH =  glm::half_pi<float>() - 0.01f;
};
