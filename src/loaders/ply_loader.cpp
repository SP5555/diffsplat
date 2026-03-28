#include "ply_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "../utils/logs.h"

/* ===== ===== PLY Parser ===== ===== */

std::vector<Gaussian3D> PLYLoader::load(const std::string &path)
{
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != "ply")
        log_fatal("PLYLoader", "Unsupported file extension: " + ext);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        log_fatal("PLYLoader", "Failed to open: " + path);

    // ===== parse header =====
    std::string line;
    int vertex_count = 0;
    std::vector<std::string> property_order;
    bool header_done = false;

    while (std::getline(file, line))
    {
        // strip trailing \r for Windows-style line endings
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.find("element vertex") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tmp;
            ss >> tmp >> tmp >> vertex_count;
        }
        else if (line.find("property float") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tmp, name;
            ss >> tmp >> tmp >> name;
            property_order.push_back(name);
        }
        else if (line == "end_header")
        {
            header_done = true;
            break;
        }
    }

    if (!header_done)
        log_fatal("PLYLoader", "Malformed PLY: end_header not found");
    if (vertex_count <= 0)
        log_fatal("PLYLoader", "Malformed PLY: no vertices found");

    // ===== build property index map =====
    std::unordered_map<std::string, int> idx;
    for (int i = 0; i < (int)property_order.size(); i++)
        idx[property_order[i]] = i;

    auto require = [&](const std::string &name) {
        if (idx.find(name) == idx.end())
            log_fatal("PLYLoader", "Missing property: " + name);
        return idx[name];
    };

    const int i_x       = require("x");
    const int i_y       = require("y");
    const int i_z       = require("z");
    const int i_scale0  = require("scale_0");
    const int i_scale1  = require("scale_1");
    const int i_scale2  = require("scale_2");
    const int i_rot0    = require("rot_0");
    const int i_rot1    = require("rot_1");
    const int i_rot2    = require("rot_2");
    const int i_rot3    = require("rot_3");
    const int i_dc0     = require("f_dc_0");
    const int i_dc1     = require("f_dc_1");
    const int i_dc2     = require("f_dc_2");
    const int i_opacity = require("opacity");

    // ===== read binary data =====
    const int stride = (int)property_order.size() * sizeof(float);
    std::vector<float> row(property_order.size());

    std::vector<Gaussian3D> splats;
    splats.reserve(vertex_count);

    for (int i = 0; i < vertex_count; i++)
    {
        file.read(reinterpret_cast<char *>(row.data()), stride);
        if (!file)
            log_fatal("PLYLoader", "Unexpected end of file at vertex " + std::to_string(i));

        Gaussian3D g;

        g.x = row[i_x];
        g.y = row[i_y];
        g.z = row[i_z];

        g.scale_x = row[i_scale0];
        g.scale_y = row[i_scale1];
        g.scale_z = row[i_scale2];

        // PLY stores (w, x, y, z) as (rot_0, rot_1, rot_2, rot_3)
        g.rot_w = row[i_rot0];
        g.rot_x = row[i_rot1];
        g.rot_y = row[i_rot2];
        g.rot_z = row[i_rot3];

        // PLY files from 3DGS training are in OpenCV convention (Z forward, Y down).
        // Convert to OpenGL convention (Z backward, Y up) by flipping Z.
        // Quaternion Z-flip: negate w and z components.
        g.z     = -g.z;
        g.rot_w = -g.rot_w;
        g.rot_z = -g.rot_z;

        // DC SH coefficients
        g.r = row[i_dc0];
        g.g = row[i_dc1];
        g.b = row[i_dc2];

        // logit-opacity stored as-is, sigmoid applied in GaussActivLayer
        g.opacity = row[i_opacity];

        splats.push_back(g);
    }

    log_info("PLYLoader", "Loaded " + std::to_string(vertex_count) + " splats from " + path);
    return splats;
}
