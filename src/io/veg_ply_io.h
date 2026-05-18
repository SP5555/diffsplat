#pragma once
#include <string>
#include <vector>
#include "../types/veg_gaussian3d.h"

/**
 * @brief Loads a VEG-format PLY file.
 *
 * VEG PLY files use element name "gaussians" (not "vertex") and store a
 * per-Gaussian scalar value instead of SH coefficients. A second element
 * "codebook_centers" may follow and is silently skipped.
 */
struct VegPLYLoader
{
    static std::vector<VegGaussian3D> load(const std::string &path);
};
