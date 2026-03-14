#include <iostream>

#include "app_plyview.h"
#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"

const int START_WIDTH = 1280;
const int START_HEIGHT = 720;

AppPlyView::AppPlyView(const std::string &plyPath, float sceneScale, CameraMode cameraMode)
    : AppBase(START_WIDTH, START_HEIGHT, "Splat viewer", true)
    , plyPath(plyPath)
    , sceneScale(sceneScale)
{
    float aspect = (float)START_WIDTH / START_HEIGHT;
    if (cameraMode == CameraMode::Arcball)
        camera = std::make_unique<ArcballCamera>(aspect);
    else if (cameraMode == CameraMode::Fly)
        camera = std::make_unique<FlyCamera>(aspect);

    std::cout << "[AppPlyView] Running:"
              << " PLY="    << plyPath
              << " Scale="  << sceneScale
              << " Camera=" << (cameraMode == CameraMode::Arcball ? "Arcball" : "Fly")
              << "\n";
}

/* ===== ===== App overrides ===== ===== */

void AppPlyView::onStart()
{
    renderer.init(width, height);
    renderer.loadPLY(plyPath, sceneScale);

    renderer.initLayers();
}

void AppPlyView::onRender()
{
    camera->update(input, dt);
    renderer.render(camera->getViewMatrix(), camera->getProjectionMatrix());
    displayFrame(renderer.getOutput());
}

void AppPlyView::onWindowResize(int newWidth, int newHeight)
{
    camera->setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
