#include "app/app_imgfit.h"
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <getopt.h>

int main(int argc, char *argv[])
{
    int width = -1;
    int height = -1;
    std::string imagePath;

    static struct option long_options[] = {
        {"width",  required_argument, 0, 'w'},
        {"height", required_argument, 0, 'h'},
        {"image",  required_argument, 0, 'i'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "w:h:i:", long_options, &idx)) != -1) {
        switch(opt) {
            case 'w': width = std::atoi(optarg); break;
            case 'h': height = std::atoi(optarg); break;
            case 'i': imagePath = optarg; break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " --width <width>"
                          << " --height <height>"
                          << " --image <path_to_image>\n";
                return 1;
        }
    }

    // Enforce tied arguments
    if ((width != -1 && height == -1) || (width == -1 && height != -1)) {
        std::cerr << "Error: width and height must be provided together.\n";
        return 1;
    }

    // Provide defaults if none were given
    if (width == -1 && height == -1) {
        width = 1280;
        height = 720;
        printf("No resolution specified, defaulting to %dx%d\n", width, height);
    }

    if (imagePath.empty()) {
        imagePath = "img/torii_moon.jpg";
        printf("No image path specified, defaulting to %s\n", imagePath.c_str());
    }

    // APP STARTS HERE
    try {
        AppImgFit app(width, height, imagePath);
        app.start();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}