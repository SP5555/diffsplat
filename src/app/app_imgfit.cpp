#include "app_imgfit.h"

/* ===== ===== Lifecycle ===== ===== */

AppImgFit::AppImgFit(int width, int height, const std::string &imagePath)
    : AppBase(width, height, "Diffsplat", false)
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
    displayFrame(renderer.getOutput());
}

void AppImgFit::onInput()
{
    // muted for now
    // if (input.mouseLeft) { ... }
    // if (input.scrollDelta != 0.f) { ... }
}
