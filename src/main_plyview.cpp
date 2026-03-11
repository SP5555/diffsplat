#include "app/app_plyview.h"
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <getopt.h>

int main(int argc, char *argv[])
{
    std::string plyPath;

    static struct option long_options[] = {
        {"scene", required_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "s:", long_options, &idx)) != -1) {
        switch (opt) {
            case 's': plyPath = optarg; break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " --scene <path_to_ply_file>\n";
                return 1;
        }
    }

    if (plyPath.empty()) {
        plyPath = "../scenes/default.ply";
        printf("No PLY path specified, defaulting to %s\n", plyPath.c_str());
    }

    try {
        AppPlyView app(plyPath);
        app.start();
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
