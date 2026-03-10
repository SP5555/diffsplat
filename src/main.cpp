#include "app/app.h"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

int main(int argc, char *argv[])
{
    int width = -1;
    int height = -1;
    std::string imagePath;

    int opt;
    while ((opt = getopt(argc, argv, "w:h:i:")) != -1) {
        switch(opt) {
            case 'w': width = std::atoi(optarg); break;
            case 'h': height = std::atoi(optarg); break;
            case 'i': imagePath = optarg; break;
            default:
                std::cerr << "Usage: " << argv[0] << " -w width -h height -i image_path\n";
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
        imagePath = "../img/torii_moon.jpg";
        printf("No image path specified, defaulting to %s\n", imagePath.c_str());
    }

    std::cout << "Running:"
              << " Width=" << width
              << " Height=" << height
              << " Image=" << imagePath << "\n";

    // APP STARTS HERE
    try {
        App app(width, height, imagePath);
        app.start();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}