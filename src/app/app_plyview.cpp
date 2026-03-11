#include "app_plyview.h"
#include <iostream>

AppPlyView::AppPlyView(const std::string &plyPath)
    : AppBase(1280, 720, "Splat viewer", true)
    , plyPath(plyPath)
{
    std::cout << "[AppPlyView] Running:"
              << " PLY=" << plyPath << "\n";
}

/* ===== ===== App overrides ===== ===== */

void AppPlyView::onStart()
{
    // TODO: load PLY, initialize renderer
}

void AppPlyView::onRender()
{
    // TODO: implement rendering

    // black screen for now
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void AppPlyView::onInput()
{
    // TODO: camera controls (orbit, pan, zoom)
    if (input.mouseLeft) {
        std::cout << "[AppPlyView] Mouse at ("
        << input.mousePos.x << ", "
        << input.mousePos.y << ")\n";
    }
}

void AppPlyView::onWindowResize(int width, int height)
{
    // TODO: rebuild projection matrix
    std::cout << "[AppPlyView] Resized to " << width << "x" << height << "\n";
}
