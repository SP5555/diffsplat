#pragma once
#include <string>
#include <memory>
#include "app_base.h"
#include "../pipelines/gauss_ply_viewer.h"
#include "../camera/camera.h"

enum class CameraMode { Arcball, Fly };

/**
 * @brief App for viewing a 3D Gaussian scene from a PLY file.
 * 
 * Supports window resizing, orbit/pan/zoom camera controls.
 * Perspective projection is pending PerspectiveProjectLayer.
 */
class AppPlyView : public AppBase
{
public:
    AppPlyView(const std::string &ply_path, float scene_scale = 1.f,
               CameraMode camera_mode = CameraMode::Fly);

protected:
    void onStart()  override;
    void onRender() override;
    void onWindowResize(int newWidth, int newHeight) override;

private:
    std::string    ply_path;
    float          scene_scale;
    GaussPlyViewer renderer;

    std::unique_ptr<Camera> camera;
};
