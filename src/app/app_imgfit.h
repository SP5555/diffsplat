#pragma once
#include <string>
#include "app_base.h"
#include "../pipelines/gauss_img_fitter.h"

/**
 * @brief App for differentiable Gaussian image fitting.
 * 
 * Owns a GaussImgFitter renderer and drives the fit loop.
 */
class AppImgFit : public AppBase
{
public:
    AppImgFit(int width, int height, const std::string &imagePath);

protected:
    void onStart()  override;
    void onRender() override;
    void onInput()  override;

private:
    std::string    imagePath;
    GaussImgFitter renderer;
};
