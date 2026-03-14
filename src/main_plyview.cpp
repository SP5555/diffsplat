#include "app/app_plyview.h"
#include <algorithm>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <unistd.h>

auto toLower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
};

int main(int argc, char *argv[])
{
    float scale = 1.f;
    std::string plyPath;
    CameraMode cameraMode = CameraMode::Arcball;

    static struct option long_options[] = {
        {"scene", required_argument, 0, 'S'},
        {"scale", required_argument, 0, 's'},
        {"camera", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "S:s:c:", long_options, &idx)) != -1) {
        switch (opt) {
            case 'S': plyPath = optarg; break;
            case 's': scale = atof(optarg); break;
            case 'c': {
                std::string cam = toLower(optarg);
                if      (cam == "fly") cameraMode = CameraMode::Fly;
                else if (cam == "arcball") cameraMode = CameraMode::Arcball;
                else {
                    std::cerr << "Invalid camera mode: " << optarg << "\n";
                    std::cerr << "Valid options: fly, arcball\n";
                    return 1;
                }
                break;
            }
            default:
                std::cerr << "Usage: " << argv[0]
                          << " --scene <path_to_ply_file>"
                          << " --scale <value>\n";
                return 1;
        }
    }

    if (plyPath.empty()) {
        plyPath = "data/ply/fly.ply";
        printf("No PLY path specified, defaulting to %s\n", plyPath.c_str());
    }

    if (scale <= 0.f) {
        printf("Invalid scale value %f, must be a positive number\n", scale);
        return 1;
    }

    try {
        AppPlyView app(plyPath, scale, cameraMode);
        app.start();
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
