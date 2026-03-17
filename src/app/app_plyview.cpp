#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "app_plyview.h"
#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"
#include "../utils/logs.h"

const int START_WIDTH = 1280;
const int START_HEIGHT = 720;

AppPlyView::AppPlyView(const std::string &ply_path, float scene_scale, CameraMode camera_mode)
    : AppBase(START_WIDTH, START_HEIGHT, "Splat viewer", true)
    , ply_path(ply_path)
    , scene_scale(scene_scale)
{
    float aspect = (float)START_WIDTH / START_HEIGHT;
    if (camera_mode == CameraMode::Arcball)
        camera = std::make_unique<ArcballCamera>(aspect);
    else if (camera_mode == CameraMode::Fly)
        camera = std::make_unique<FlyCamera>(aspect);

    log_info("AppPlyView",
        "PLY=" + ply_path +
        " Scale=" + std::to_string(scene_scale) +
        " Camera=" + (camera_mode == CameraMode::Arcball ? "Arcball" : "Fly")
    );
}

/* ===== ===== App overrides ===== ===== */

void AppPlyView::onStart()
{
    renderer.init(width, height);
    renderer.loadPLY(ply_path, scene_scale);

    renderer.initLayers();
}

void AppPlyView::onFrame()
{
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse && !io.WantCaptureKeyboard)
        camera->update(input, dt);

    renderer.render(camera->getViewMatrix(), camera->getProjectionMatrix());
    displayFrame(renderer.getOutput());

    // ImGui
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(220, 100), ImGuiCond_Once);
    ImGui::Begin("Splat Viewer");

    ImGui::Text("FPS: %.2f", getFPS());
    ImGui::Text("Visible Splats: %d", renderer.getVisibleCount());
    auto pos = camera->getPosition();
    ImGui::Text("Camera: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);

    ImGui::End();
}

void AppPlyView::onWindowResize(int newWidth, int newHeight)
{
    camera->setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
