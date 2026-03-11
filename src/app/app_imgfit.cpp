#include <iostream>

#include "app_imgfit.h"

AppImgFit::AppImgFit(int width, int height, const std::string &imagePath)
    : AppBase(width, height, "Image Fitter", false)
    , imagePath(imagePath)
{
    std::cout << "[AppImgFit] Running:"
              << " Width=" << width
              << " Height=" << height
              << " Image=" << imagePath << "\n";
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
    // if (input.mouseLeft) {
    //     std::cout << "[AppImgFit] Mouse at ("
    //               << input.mousePos.x << ", "
    //               << input.mousePos.y << ")\n";
    // }
}
