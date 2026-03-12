#include "app/app_plyview.h"
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <getopt.h>

int main(int argc, char *argv[])
{
    float scale = 1.f;
    std::string plyPath;

    static struct option long_options[] = {
        {"scene", required_argument, 0, 's'},
        {"scale", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "s:c:", long_options, &idx)) != -1) {
        switch (opt) {
            case 's': plyPath = optarg; break;
            case 'c': scale = atof(optarg); break;
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
        AppPlyView app(plyPath, scale);
        app.start();
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
