#pragma once
#include <memory>
#include <string>
#include <vector>

#include "app_base.h"
#include "../pipelines/veg_renderer.h"
#include "../camera/camera.h"

// TF widget pulls in GLFW; include guard avoids double-include with glad
#define GLFW_INCLUDE_NONE
#include "../../../third_party/tfn/widget.h"

enum class VegCameraMode { Arcball, Fly };

/**
 * @brief Interactive VEG volume viewer.
 *
 * Renders a VEG PLY file with a user-controlled transfer function widget.
 * The TF widget fires a callback on every edit; the callback uploads a new
 * 256-entry LUT to the GPU and triggers a re-render on the next frame.
 */
class VegApp : public AppBase
{
public:
    VegApp(const std::string &ply_path, float scene_scale = 1.f,
           VegCameraMode camera_mode = VegCameraMode::Fly);

protected:
    void onStart()  override;
    void onFrame()  override;
    void onWindowResize(int newWidth, int newHeight) override;

private:
    void loadScene(const std::string &path);
    void buildLutFromWidget(const tfn::list3f &colors,
                            const tfn::list2f &alphas);

    std::string    ply_path;
    float          scene_scale;
    VegCameraMode  camera_mode;
    VegRenderer    renderer;

    std::unique_ptr<Camera> camera;
    std::unique_ptr<tfn::TransferFunctionWidget> tf_widget;

    // cached LUT from the last TF widget callback; re-applied on scene reload
    std::vector<float> cached_lut_rgb;
    std::vector<float> cached_lut_alpha;

    // FPS gadget (same pattern as ViewerApp)
    std::vector<float> fps_history;
    std::vector<float> x_axis;
    int         frame_count = 0;
    std::string load_error;
};
