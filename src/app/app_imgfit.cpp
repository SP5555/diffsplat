#include <iostream>

#include "app_imgfit.h"

AppImgFit::AppImgFit(int width, int height, const std::string &imagePath, int splatCount)
    : AppBase(width, height, "Image Fitter", false)
    , imagePath(imagePath)
    , splatCount(splatCount)
{
    std::cout << "[AppImgFit] Running:"
              << " Width="      << width
              << " Height="     << height
              << " Image="      << imagePath
              << " SplatCount=" << splatCount << "\n";
}

/* ===== ===== App overrides ===== ===== */

void AppImgFit::onStart()
{
    renderer.init(width, height);
    renderer.loadTargetImage(imagePath, width, height, 10);
    renderer.randomInitGaussians(splatCount);
    // layers can only be wired after gaussians are initialized
    // as it needs to know the gaussian count for allocation
    renderer.initLayers();
}

void AppImgFit::onRender()
{
    renderer.render();
    displayFrame(renderer.getOutput());
}
