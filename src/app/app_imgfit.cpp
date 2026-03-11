#include "app_imgfit.h"

/* ===== ===== Lifecycle ===== ===== */

AppImgFit::AppImgFit(int width, int height, const std::string &imagePath)
    : AppBase(width, height, "Diffsplat")
    , imagePath(imagePath)
{
}

/* ===== ===== App overrides ===== ===== */

void AppImgFit::onStart()
{
    renderer.init(width, height);
    renderer.loadTargetImage(imagePath, width, height, 10);
    renderer.randomInitGaussians(80000);
}

void AppImgFit::onRender()
{
    renderer.render();
}

void AppImgFit::onInput()
{
    // muted for now
    // if (input.mouseLeft)
    // {
    //     std::cout << "[App] Mouse Left Held at (" << input.mousePos.x << ", " << input.mousePos.y << ")\n";
    // }
    // if (input.mouseRight)
    // {
    //     std::cout << "[App] Mouse Right Held at (" << input.mousePos.x << ", " << input.mousePos.y << ")\n";
    // }
    // if (input.scrollDelta != 0.f)
    // {
    //     std::cout << "[App] Scroll Delta: " << input.scrollDelta << "\n";
    // }
}
