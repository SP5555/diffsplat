#include <iostream>

#include "app_plyview.h"
#include "../camera/camera.h"

const int START_WIDTH = 1280;
const int START_HEIGHT = 720;

AppPlyView::AppPlyView(const std::string &plyPath)
    : AppBase(START_WIDTH, START_HEIGHT, "Splat viewer", true)
    , plyPath(plyPath)
    , camera((float)START_WIDTH / START_HEIGHT)
{
    std::cout << "[AppPlyView] Running:"
              << " PLY=" << plyPath << "\n";
}

/* ===== ===== App overrides ===== ===== */

void AppPlyView::onStart()
{
    renderer.init(width, height);
    renderer.loadPLY(plyPath);

    renderer.initLayers();
}

void AppPlyView::onRender()
{
    renderer.render(/* camera.getViewMatrix(), camera.getProjectionMatrix() */);
    // displayFrame(renderer.getOutput());
}

void AppPlyView::onInput()
{
    // TODO: pass dt once AppBase exposes it
    camera.update(input.mouseDelta, input.scrollDelta, input.shiftPressed, dt);

    if (input.mouseLeftPressed) {
        printf("[AppPlyView] Mouse delta : (%.2f, %.2f)\n", input.mouseDelta.x, input.mouseDelta.y);
    }
}

void AppPlyView::onWindowResize(int newWidth, int newHeight)
{
    camera.setAspect((float)newWidth / newHeight);
    std::cout << "[AppPlyView] Resized to " << newWidth << "x" << newHeight << "\n";
}
