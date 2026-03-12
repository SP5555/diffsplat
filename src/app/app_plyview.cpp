#include <iostream>

#include "app_plyview.h"
#include "../camera/camera.h"

const int START_WIDTH = 1280;
const int START_HEIGHT = 720;

AppPlyView::AppPlyView(const std::string &plyPath, float sceneScale)
    : AppBase(START_WIDTH, START_HEIGHT, "Splat viewer", true)
    , plyPath(plyPath)
    , sceneScale(sceneScale)
    , camera((float)START_WIDTH / START_HEIGHT)
{
    std::cout << "[AppPlyView] Running:"
              << " PLY="    << plyPath
              << " Scale="  << sceneScale << "\n";
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
    camera.update(input, dt);
    renderer.render(camera.getViewMatrix(), camera.getProjectionMatrix());
    displayFrame(renderer.getOutput());
}

void AppPlyView::onWindowResize(int newWidth, int newHeight)
{
    camera.setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
