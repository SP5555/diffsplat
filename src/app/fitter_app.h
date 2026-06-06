#pragma once
#include <string>
#include <vector>
#include "app_base.h"
#include "../pipelines/image_fitter.h" // also brings in MCMCConfig

// App for differentiable Gaussian image fitting.
// Owns an ImageFitter pipeline and drives the fit loop.
class FitterApp : public AppBase
{
public:
    FitterApp(int width, int height, const std::string &image_path,
              int splat_count = 60000, const MCMCConfig& mcmc_cfg = {});

protected:
    void onStart()  override;
    void onFrame() override;

private:
    int getIterCount() const { return fitter.getIterCount(); }
    float getLoss()    const { return fitter.getLoss(); }

    std::string  image_path;
    int          splat_count;
    MCMCConfig   mcmc_cfg;
    ImageFitter  fitter;

    // ImGui
    std::vector<float> loss_history;
    std::vector<float> iter_history;
    char save_ply_path[256] = "output.ply";
};
