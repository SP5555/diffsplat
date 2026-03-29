#include "image_saver.h"

#include <cuda_runtime.h>
#include <stdint.h>

#include "../cuda/cuda_check.h"

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void ImageSaver::saveAsPNG(const float *d_pixels, int width, int height, const std::string &path)
{
    std::vector<float> h_pixels(width * height * 3);
    CUDA_CHECK(cudaMemcpy(h_pixels.data(), d_pixels, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    // convert float [0.0, 1.0] to uint8 [0, 255]
    std::vector<uint8_t> img_data(width * height * 3);
    for (int i = 0; i < width * height * 3; i++)
        img_data[i] = (uint8_t)(fminf(fmaxf(h_pixels[i] * 255.f, 0.f), 255.f));

    stbi_write_png(path.c_str(), width, height, 3, img_data.data(), width * 3);
}