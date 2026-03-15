#include <iostream>

#include "app_imgfit.h"

AppImgFit::AppImgFit(int width, int height, const std::string &image_path, int splat_count)
    : AppBase(width, height, "Image Fitter", false)
    , image_path(image_path)
    , splat_count(splat_count)
{
    std::cout << "[AppImgFit] Running:"
              << " Width="      << width
              << " Height="     << height
              << " Image="      << image_path
              << " SplatCount=" << splat_count << "\n";
}

/* ===== ===== App overrides ===== ===== */

void AppImgFit::onStart()
{
    fitter.init(width, height);
    fitter.loadTargetImage(image_path, width, height, 10);
    fitter.randomInitGaussians(splat_count);
    // layers can only be wired after gaussians are initialized
    // as it needs to know the gaussian count for allocation
    fitter.initLayers();
}

void AppImgFit::onRender()
{
    fitter.render();
    displayFrame(fitter.getOutput());

    char buf[128];
    sprintf(buf, "Image Fitter [Iter: %d] [Loss: %.8f]",
            fitter.getIterCount(), fitter.getLoss());
    updateWindowTitle(buf);
}
