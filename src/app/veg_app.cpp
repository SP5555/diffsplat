#include "veg_app.h"

#include <algorithm>
#include <numeric>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "tinyfiledialogs.h"

#include <cuda_runtime.h>

#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"
#include "../utils/logs.h"

static constexpr int START_WIDTH  = 1280;
static constexpr int START_HEIGHT = 720;
static constexpr int GRAPH_HISTORY_SIZE       = 100;
static constexpr int UPDATE_FPS_EVERY_N_FRAMES = 2;

VegApp::VegApp(const std::string &ply_path, float scene_scale, VegCameraMode camera_mode)
    : AppBase(START_WIDTH, START_HEIGHT, "VEG Viewer", true)
    , ply_path(ply_path)
    , scene_scale(scene_scale)
    , camera_mode(camera_mode)
{
    float aspect = (float)START_WIDTH / START_HEIGHT;
    if (camera_mode == VegCameraMode::Arcball)
        camera = std::make_unique<ArcballCamera>(aspect);
    else
        camera = std::make_unique<FlyCamera>(aspect);

    x_axis.resize(GRAPH_HISTORY_SIZE);
    std::iota(x_axis.begin(), x_axis.end(), 0.f);
}

/* ===== ===== Helpers ===== ===== */

void VegApp::buildLutFromWidget(const tfn::list3f &colors, const tfn::list2f &alphas)
{
    // colors[i] = {r, g, b},  alphas[i] = {position, opacity}
    // Both are uniformly sampled at VEG_LUT_SIZE entries by the widget.
    int n = (int)colors.size();
    if (n == 0) return;

    cached_lut_rgb.resize(n * 3);
    cached_lut_alpha.resize(n);
    for (int i = 0; i < n; i++)
    {
        cached_lut_rgb[i*3+0] = colors[i].x;
        cached_lut_rgb[i*3+1] = colors[i].y;
        cached_lut_rgb[i*3+2] = colors[i].z;
        cached_lut_alpha[i]   = alphas[i].y; // .x = position (0..1), .y = opacity
    }
    renderer.setLUT(cached_lut_rgb.data(), cached_lut_alpha.data());
}

void VegApp::loadScene(const std::string &path)
{
    load_error.clear();
    try {
        renderer.reloadPLY(path, scene_scale);
        // re-push the widget's current LUT now that the GPU buffers are allocated
        if (!cached_lut_rgb.empty())
            renderer.setLUT(cached_lut_rgb.data(), cached_lut_alpha.data());
    } catch (const std::exception &e) {
        load_error = e.what();
        log_error("VegApp", "Failed to load: " + load_error);
        cudaGetLastError();
    }
}

/* ===== ===== App overrides ===== ===== */

void VegApp::onStart()
{
    renderer.init(width, height);

    // Build the TF widget after GL is initialised (it creates a GL texture).
    tf_widget = std::make_unique<tfn::TransferFunctionWidget>(
        [this](const tfn::list3f &colors,
               const tfn::list2f &alphas,
               const tfn::vec2f  &/*valueRange*/)
        {
            buildLutFromWidget(colors, alphas);
        }
    );

    if (!ply_path.empty())
        loadScene(ply_path);
}

void VegApp::onFrame()
{
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse && !io.WantCaptureKeyboard)
        camera->update(input, dt);

    // ===== Transfer function widget =====
    // Must come before renderer.render() so the LUT is current this frame.
    ImGui::SetNextWindowPos(ImVec2(2, 230), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(400, 340), ImGuiCond_Once);
    ImGui::Begin("Transfer Function");
    tf_widget->build_gui();
    ImGui::Dummy(ImVec2(0, 0));
    ImGui::End();
    tf_widget->render(); // fires callback -> cudaMemcpy if TF changed

    if (renderer.isLoaded())
    {
        renderer.render(camera->getViewMatrix(), camera->getProjectionMatrix());
        displayFrame(renderer.getOutput());
    }

    // FPS history
    frame_count++;
    if (renderer.isLoaded() && frame_count % UPDATE_FPS_EVERY_N_FRAMES == 0)
    {
        fps_history.push_back(getFPS());
        if ((int)fps_history.size() > GRAPH_HISTORY_SIZE)
            fps_history.erase(fps_history.begin());
    }

    // ===== Left control panel =====
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(260, 220), ImGuiCond_Once);
    ImGui::Begin("VEG Viewer");

    if (ImGui::Button("Open PLY..."))
    {
        const char *filters[] = {"*.ply"};
        const char *result = tinyfd_openFileDialog("Open VEG PLY", "", 1, filters, "PLY files", 0);
        if (result)
        {
            ply_path = result;
            loadScene(ply_path);
            fps_history.clear();
        }
    }

    if (!load_error.empty())
    {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.4f, 1.f));
        ImGui::TextWrapped("Error: %s", load_error.c_str());
        ImGui::PopStyleColor();
    }

    if (renderer.isLoaded())
    {
        ImGui::Text("Visible Splats: %d", renderer.getVisibleCount());
        auto pos = camera->getPosition();
        ImGui::Text("Camera: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
        ImGui::Text("FPS: %.2f  |  %.2f ms", getFPS(), getFrametime());

        int history_size = (int)fps_history.size();
        if (history_size > 0 &&
            ImPlot::BeginPlot("##FPS", ImVec2(-1, ImGui::GetTextLineHeight() * 8)))
        {
            ImPlot::SetupAxes(nullptr, "FPS", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, GRAPH_HISTORY_SIZE, ImPlotCond_Always);
            float max_fps = *std::max_element(fps_history.begin(), fps_history.end());
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_fps + 20, ImPlotCond_Always);

            ImPlotSpec spec;
            spec.LineColor  = ImVec4(0.4f, 0.9f, 0.4f, 1.f);
            spec.LineWeight = 2.f;
            ImPlot::PlotLine("FPS", x_axis.data(), fps_history.data(), history_size, spec);
            ImPlot::EndPlot();
        }
    }

    ImGui::End();

}

void VegApp::onWindowResize(int newWidth, int newHeight)
{
    camera->setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
