#pragma once
#include <vector>
#include "../types/gaussian3d.h"

static constexpr float C0 = 0.28209f;  // DC SH coefficient

namespace SplatUtils
{
    std::vector<Gaussian3D> randomInit(int count, int width, int height, int seed)
    {
        srand(seed);
        auto rnd  = []() { return ((float)rand() / RAND_MAX) * 2.f - 1.f; };
        auto rndu = []() { return  (float)rand() / RAND_MAX; };

        float half_w    = (float)width  * 0.5f;
        float half_h    = (float)height * 0.5f;
        float log_sigma = logf(3.f);

        std::vector<Gaussian3D> splats(count);
        for (auto &g : splats)
        {
            g.x = rnd() * half_w;
            g.y = rnd() * half_h;
            g.z = rnd();

            g.scale_x = log_sigma + rnd() * 0.5f;
            g.scale_y = log_sigma + rnd() * 0.5f;
            g.scale_z = log_sigma + rnd() * 0.5f;

            g.rot_w = 1.f + rnd() * 0.1f;
            g.rot_x =       rnd() * 0.1f;
            g.rot_y =       rnd() * 0.1f;
            g.rot_z =       rnd() * 0.1f;
            float norm = sqrtf(g.rot_w*g.rot_w + g.rot_x*g.rot_x +
                            g.rot_y*g.rot_y + g.rot_z*g.rot_z);
            g.rot_w /= norm; g.rot_x /= norm;
            g.rot_y /= norm; g.rot_z /= norm;

            // initialize DC SH coefficients to random colors in [0, 1]
            g.r = (rndu() - 0.5f) / C0;
            g.g = (rndu() - 0.5f) / C0;
            g.b = (rndu() - 0.5f) / C0;

            float o_raw = 0.6f + 0.4f * rndu();
            g.opacity = logf(o_raw / (1.f - o_raw));
        }

        return splats;
    }
    
    void normalizeScene(std::vector<Gaussian3D> &splats, float sceneScale = 1.f)
    {
        // compute centroid
        float cx = 0.f, cy = 0.f, cz = 0.f;
        for (const auto &g : splats)
        {
            cx += g.x;
            cy += g.y;
            cz += g.z;
        }
        float inv = 1.f / (float)splats.size();
        cx *= inv; cy *= inv; cz *= inv;

        // translate to centroid, find max extent
        float maxExt = 0.f;
        for (auto &g : splats)
        {
            g.x -= cx; g.y -= cy; g.z -= cz;

            // OpenCV -> OpenGL convention
            // flip along any one axis to fix left-handedness
            // here we flip Z. Flipping Y would make the world upside down.
            g.z      = -g.z;
            g.rot_w  = -g.rot_w;
            g.rot_z  = -g.rot_z;

            maxExt = std::max(maxExt, std::abs(g.x));
            maxExt = std::max(maxExt, std::abs(g.y));
            maxExt = std::max(maxExt, std::abs(g.z));
        }

        // scale to [-1, 1]
        if (maxExt > 0.f)
        {
            float linearScale = sceneScale / maxExt;
            float logScale = logf(linearScale);
            for (auto &g : splats)
            {
                g.x *= linearScale;
                g.y *= linearScale;
                g.z *= linearScale;

                g.scale_x += logScale;
                g.scale_y += logScale;
                g.scale_z += logScale;
            }
        }
    }
}
