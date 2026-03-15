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
    AppImgFit(int width, int height, const std::string &image_path, int splat_count = 60000);

protected:
    void onStart()  override;
    void onRender() override;

private:
    int getIterCount() const { return fitter.getIterCount(); }
    float getLoss()    const { return fitter.getLoss(); }

    std::string     image_path;
    int             splat_count;
    GaussImgFitter  fitter;
};
