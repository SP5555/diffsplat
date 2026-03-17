#include <algorithm>
#include <cstring>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include "image_loader.h"
#include "../utils/logs.h"

LoadedImage ImageLoader::load(const std::string &path, int target_w, int target_h, int padding)
{
    int src_w, src_h, channels;
    unsigned char *data = stbi_load(path.c_str(), &src_w, &src_h, &channels, 3);
    if (!data)
    {
        log_error("ImageLoader", "Failed to load image: " + path);
        log_error("ImageLoader", "Reason: " + std::string(stbi_failure_reason()));
        return {};
    }

    // scale to fit inside canvas preserving aspect ratio
    float scale = std::min(
        (float)(target_w - 2 * padding) / src_w,
        (float)(target_h - 2 * padding) / src_h
    );

    int fit_w = (int)(src_w * scale);
    int fit_h = (int)(src_h * scale);

    std::vector<unsigned char> resized(fit_w * fit_h * 3);
    stbir_resize_uint8_linear(
        data, src_w, src_h, 0,
        resized.data(), fit_w, fit_h, 0,
        STBIR_RGB
    );

    stbi_image_free(data);

    // place centered on canvas, fill outside with black
    std::vector<float> pixels(target_w * target_h * 3, 0.f);

    int offset_x = (target_w - fit_w) / 2;
    int offset_y = (target_h - fit_h) / 2;

    for (int y = 0; y < fit_h; y++)
    {
        for (int x = 0; x < fit_w; x++)
        {
            int src_idx = (y * fit_w + x) * 3;
            int dst_idx = ((y + offset_y) * target_w + (x + offset_x)) * 3;
            pixels[dst_idx + 0] = resized[src_idx + 0] / 255.f;
            pixels[dst_idx + 1] = resized[src_idx + 1] / 255.f;
            pixels[dst_idx + 2] = resized[src_idx + 2] / 255.f;
        }
    }

    log_info("ImageLoader",
        "Loaded " + path +
        " (" + std::to_string(src_w) + "x" + std::to_string(src_h) + ")" +
        " -> canvas " + std::to_string(target_w) + "x" + std::to_string(target_h) +
        ", fit " + std::to_string(fit_w) + "x" + std::to_string(fit_h)
    );

    return {pixels, target_w, target_h, src_w, src_h};
}