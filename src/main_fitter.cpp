#include <iostream>
#include "cxxopts.hpp"
#include "app/fitter_app.h"
#include "density/mcmc_config.h"
#include "utils/logs.h"

int main(int argc, char *argv[])
{
    cxxopts::Options options("fitter", "Image Fitter");

    options.add_options()
        ("w,width",        "Window width",                  cxxopts::value<int>()->default_value("1280"))
        ("h,height",       "Window height",                 cxxopts::value<int>()->default_value("720"))
        ("i,image",        "Path to target image",          cxxopts::value<std::string>()->default_value("data/img/torii_moon.jpg"))
        ("s,splat-count",  "Number of starting splats",     cxxopts::value<int>()->default_value("60000"))
        // MCMC density control (defaults live in MCMCConfig in mcmc_config.h)
        ("density",     "Enable MCMC density control")
        ("warmup",      "Steps before density control starts",          cxxopts::value<int>())
        ("dc-interval", "Density control interval in steps",            cxxopts::value<int>())
        ("opacity-reg", "L1 opacity regularization strength (per step)",cxxopts::value<float>())
        ("dead-thresh", "Opacity threshold to trigger relocation",      cxxopts::value<float>())
        ("reloc-cap",   "Max fraction relocated per pass (0-1)",        cxxopts::value<float>())
        ("perturb-std",   "Positional noise std for relocated Gaussians", cxxopts::value<float>())
        ("split-factor",  "Opacity split for parent and child on relocation", cxxopts::value<float>())
        ("pos-noise",   "Brownian position noise std per step (world units)", cxxopts::value<float>())
        ("scale-noise", "Brownian log-scale noise std per step",             cxxopts::value<float>())
        ("help",        "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int width              = result["width"].as<int>();
    int height             = result["height"].as<int>();
    std::string image_path = result["image"].as<std::string>();
    int splat_count        = result["splat-count"].as<int>();

    // Start from struct defaults; only override fields the user explicitly passed.
    MCMCConfig mcmc_cfg;
    mcmc_cfg.enabled = result.count("density") > 0;
    if (result.count("warmup"))      mcmc_cfg.warmup_steps       = result["warmup"].as<int>();
    if (result.count("dc-interval")) mcmc_cfg.interval           = result["dc-interval"].as<int>();
    if (result.count("opacity-reg")) mcmc_cfg.opacity_reg_lambda = result["opacity-reg"].as<float>();
    if (result.count("dead-thresh")) mcmc_cfg.dead_threshold     = result["dead-thresh"].as<float>();
    if (result.count("reloc-cap"))   mcmc_cfg.max_reloc_fraction = result["reloc-cap"].as<float>();
    if (result.count("perturb-std"))  mcmc_cfg.perturb_std        = result["perturb-std"].as<float>();
    if (result.count("split-factor")) mcmc_cfg.split_factor        = result["split-factor"].as<float>();
    if (result.count("pos-noise"))    mcmc_cfg.pos_noise_std        = result["pos-noise"].as<float>();
    if (result.count("scale-noise"))  mcmc_cfg.scale_noise_std      = result["scale-noise"].as<float>();

    /* ===== Argument Validation ===== */
    if ((result.count("width") && !result.count("height")) ||
        (!result.count("width") && result.count("height"))) {
        log_error("Main", "Error: --width and --height must be specified together.");
        return 1;
    }

    if (width <= 0 || height <= 0) {
        log_error("Main", "Error: width and height must be positive integers.");
        return 1;
    }

    if (!result.count("width") && !result.count("height")) {
        char buf[128];
        snprintf(buf, sizeof(buf), "No width/height specified, defaulting to %dx%d", width, height);
        log_info("Main", buf);
    }

    if (splat_count <= 0) {
        log_error("Main", "Error: splat count must be positive.");
        return 1;
    }

    try {
        FitterApp app(width, height, image_path, splat_count, mcmc_cfg);
        app.start();
    }
    catch (const std::exception &e)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Failed to start application: %s", e.what());
        log_error("Main", buf);
        return 1;
    }

    return 0;
}