#pragma once
#include <string>

/**
 * @brief saves a float RGB buffer to a PNG file on disk.
 * 
 */
class ImageSaver
{
public:
    static void saveAsPNG(const float *d_pixels, int width, int height, const std::string &path);
};
