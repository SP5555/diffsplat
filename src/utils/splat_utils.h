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

            float s = rnd() * 0.5;
            g.scale_x = log_sigma + s;
            g.scale_y = log_sigma + s;
            g.scale_z = log_sigma + s;

            g.rot_w = 1.f;
            g.rot_x = 0.f;
            g.rot_y = 0.f;
            g.rot_z = 0.f;
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
    
    void normalizeScene(std::vector<Gaussian3D> &splats, float scene_scale = 1.f)
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
        float max_ext = 0.f;
        for (auto &g : splats)
        {
            g.x -= cx; g.y -= cy; g.z -= cz;

            // OpenCV -> OpenGL convention
            // flip along any one axis to fix left-handedness
            // here we flip Z. Flipping Y would make the world upside down.
            g.z      = -g.z;
            g.rot_w  = -g.rot_w;
            g.rot_z  = -g.rot_z;

            max_ext = std::max(max_ext, std::abs(g.x));
            max_ext = std::max(max_ext, std::abs(g.y));
            max_ext = std::max(max_ext, std::abs(g.z));
        }

        // scale to [-1, 1]
        if (max_ext > 0.f)
        {
            float linear_scale = scene_scale / max_ext;
            float log_scale = logf(linear_scale);
            for (auto &g : splats)
            {
                g.x *= linear_scale;
                g.y *= linear_scale;
                g.z *= linear_scale;

                g.scale_x += log_scale;
                g.scale_y += log_scale;
                g.scale_z += log_scale;
            }
        }
    }
}
