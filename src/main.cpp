#include "app/app.h"
#include <iostream>

int main(int argc, char *argv[])
{
    // get app resolution from command line args (optional)
    int width = 1280;
    int height = 720;
    if (argc >= 3)    {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    }

    std::string imagePath = "../img/torii_moon.jpg";
    if (argc >= 4) {
        imagePath = argv[3];
    }

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