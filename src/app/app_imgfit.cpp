#include <algorithm>
#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "app_imgfit.h"
#include "../utils/logs.h"

// ImPlot
const int GRAPH_HISTORY_SIZE = 100;
const int UPDATE_FPS_EVERY_N_FRAMES = 5;

AppImgFit::AppImgFit(int width, int height, const std::string &image_path, int splat_count)
    : AppBase(width, height, "Image Fitter", false)
    , image_path(image_path)
    , splat_count(splat_count)
{
    log_info("AppImgFit",
        "Width=" + std::to_string(width) +
        " Height=" + std::to_string(height) +
        " Image=" + image_path +
        " SplatCount=" + std::to_string(splat_count)
    );
}

/* ===== ===== App overrides ===== ===== */

void AppImgFit::onStart()
{
    fitter.init(width, height);
    fitter.loadTargetImage(image_path, width, height, 10);
    fitter.randomInitGaussians(splat_count);
    // layers can only be wired after gaussians are initialized
    // as it needs to know the gaussian count for allocation
    fitter.initLayers();
}

void AppImgFit::onFrame()
{
    fitter.step();
    displayFrame(fitter.getOutput());

    float current_loss = getLoss();

    static int frame_count = 0;
    frame_count++;
    if (frame_count % UPDATE_FPS_EVERY_N_FRAMES == 0) {
        loss_history.push_back(current_loss);
        iter_history.push_back(static_cast<float>(getIterCount()));
        if (loss_history.size() > GRAPH_HISTORY_SIZE) {
            loss_history.erase(loss_history.begin());
            iter_history.erase(iter_history.begin());
        }
    }

    // ImGui
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(240, 260), ImGuiCond_Once);
    ImGui::Begin("Image Fitter");

    ImGui::Text("FPS: %.2f", getFPS());
    ImGui::Text("Iteration: %d", getIterCount());
    ImGui::Text("Loss: %.8f", current_loss);

    int history_size = static_cast<int>(loss_history.size());
    if (history_size > 0 && ImPlot::BeginPlot("##LossRolling", ImVec2(-1, ImGui::GetTextLineHeight() * 12))) {

        ImPlot::SetupAxes("Iteration", "Loss", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);

        // X axis
        float x_min = iter_history.front();
        float x_max = iter_history.back();
        ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImPlotCond_Always);

        // Y axis
        float min_loss = *std::min_element(loss_history.begin(), loss_history.end());
        float max_loss = *std::max_element(loss_history.begin(), loss_history.end());
        if (min_loss == max_loss) max_loss += 1e-6f;
        float padding = (max_loss - min_loss) * 0.2f;
        ImPlot::SetupAxisLimits(ImAxis_Y1, min_loss - padding, max_loss + padding, ImPlotCond_Always);

        ImPlotSpec spec;
        spec.LineColor = ImVec4(0.9f, 0.9f, 0.4f, 1.0f);
        spec.LineWeight = 2.0f;
        ImPlot::PlotLine("Loss", iter_history.data(), loss_history.data(), history_size, spec);

        ImPlot::EndPlot();
    }

    ImGui::End();
}
