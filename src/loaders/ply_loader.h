#pragma once
#include <string>
#include <vector>

#include "../types/gaussian3d.h"

/**
 * @brief Loads a 3DGS PLY file into a vector of Gaussian3D structs.
 * 
 * Expects binary little-endian PLY format as output by standard
 * 3D Gaussian Splatting training (e.g. gaussian-splatting, gsplat).
 * 
 * Conversions applied:
 *   - Color: f_dc * C0 + 0.5  (DC spherical harmonic -> linear RGB)
 *   - Scale: stored as-is     (already log-scale, exp applied in GaussActivLayer)
 *   - Opacity: stored as-is   (already logit, sigmoid applied in GaussActivLayer)
 *   - Rotation: rot_0 = w, rot_1 = x, rot_2 = y, rot_3 = z
 * 
 * Higher-order SH coefficients (f_rest_*) are ignored for now.
 */
class PLYLoader
{
public:
    static std::vector<Gaussian3D> load(const std::string &path);
};
