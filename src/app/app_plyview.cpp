#include <algorithm>
#include <iostream>
#include <numeric>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "app_plyview.h"
#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"
#include "../utils/logs.h"

const int START_WIDTH = 1280;
const int START_HEIGHT = 720;

// ImPlot
const int GRAPH_HISTORY_SIZE = 100;
const int UPDATE_FPS_EVERY_N_FRAMES = 2;

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

    x_axis.resize(GRAPH_HISTORY_SIZE);
    std::iota(x_axis.begin(), x_axis.end(), 0.f);
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

    static int frame_count = 0;
    frame_count++;
    if (frame_count % UPDATE_FPS_EVERY_N_FRAMES == 0) {
        fps_history.push_back(getFPS());
        if (fps_history.size() > GRAPH_HISTORY_SIZE)
            fps_history.erase(fps_history.begin());
    }

    // ImGui
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(240, 200), ImGuiCond_Once);
    ImGui::Begin("Splat Viewer");

    ImGui::Text("Visible Splats: %d", renderer.getVisibleCount());
    auto pos = camera->getPosition();
    ImGui::Text("Camera: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
    ImGui::Text("FPS: %.2f\t| Frametime: %.2f ms", getFPS(), getFrametime());

    int history_size = static_cast<int>(fps_history.size());
    if (history_size > 0 && ImPlot::BeginPlot("##FPSRolling", ImVec2(-1, ImGui::GetTextLineHeight() * 8))) {

        ImPlot::SetupAxes(nullptr, "FPS", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);

        // X axis
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, GRAPH_HISTORY_SIZE, ImPlotCond_Always);

        // Y axis
        float max_fps = *std::max_element(fps_history.begin(), fps_history.end());
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_fps + 20, ImPlotCond_Always);

        ImPlotSpec spec;
        spec.LineColor = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
        spec.LineWeight = 2.0f;
        ImPlot::PlotLine("FPS", x_axis.data(), fps_history.data(), history_size, spec);

        ImPlot::EndPlot();
    }

    ImGui::End();
}

void AppPlyView::onWindowResize(int newWidth, int newHeight)
{
    camera->setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
