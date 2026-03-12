#pragma once
#include <string>
#include "app_base.h"
#include "../pipelines/gauss_ply_viewer.h"
#include "../camera/camera.h"

/**
 * @brief App for viewing a 3D Gaussian scene from a PLY file.
 * 
 * Supports window resizing, orbit/pan/zoom camera controls.
 * Perspective projection is pending PerspectiveProjectLayer.
 */
class AppPlyView : public AppBase
{
public:
    AppPlyView(const std::string &plyPath);

protected:
    void onStart()  override;
    void onRender() override;
    void onInput()  override;
    void onWindowResize(int newWidth, int newHeight) override;

private:
    std::string      plyPath;
    GaussPlyViewer   renderer;
    Camera           camera;
};
