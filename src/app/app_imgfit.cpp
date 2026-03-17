#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "app_imgfit.h"
#include "../utils/logs.h"

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

    // ImGui
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(160, 100), ImGuiCond_Once);
    ImGui::Begin("Image Fitter");

    ImGui::Text("FPS: %.2f", getFPS());
    ImGui::Text("Iteration: %d", getIterCount());
    ImGui::Text("Loss: %.8f", getLoss());

    ImGui::End();
}
