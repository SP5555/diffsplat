#include <iostream>

#include "app_plyview.h"
#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"

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

    std::cout << "[AppPlyView] Running:"
              << " PLY="    << ply_path
              << " Scale="  << scene_scale
              << " Camera=" << (camera_mode == CameraMode::Arcball ? "Arcball" : "Fly")
              << "\n";
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
    camera->update(input, dt);
    renderer.render(camera->getViewMatrix(), camera->getProjectionMatrix());
    displayFrame(renderer.getOutput());

    char buf[128];
    auto pos = camera->getPosition();
    sprintf(buf, "Splat Viewer [Visible Splats: %d] [Camera Pos: (%.4f, %.4f, %.4f)]",
        renderer.getVisibleCount(), pos.x, pos.y, pos.z);
    updateWindowTitle(buf);
}

void AppPlyView::onWindowResize(int newWidth, int newHeight)
{
    camera->setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
