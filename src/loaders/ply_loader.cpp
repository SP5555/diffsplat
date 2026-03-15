#include "ply_loader.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
#include <cstring>

/* ===== ===== PLY Parser ===== ===== */

std::vector<Gaussian3D> PLYLoader::load(const std::string &path)
{
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != "ply")
        throw std::runtime_error("[PLYLoader] Unsupported file extension: " + ext);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("[PLYLoader] Failed to open: " + path);

    // ===== parse header =====
    std::string line;
    int vertexCount = 0;
    std::vector<std::string> propertyOrder;
    bool headerDone = false;

    while (std::getline(file, line))
    {
        // strip trailing \r for Windows-style line endings
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.find("element vertex") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tmp;
            ss >> tmp >> tmp >> vertexCount;
        }
        else if (line.find("property float") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tmp, name;
            ss >> tmp >> tmp >> name;
            propertyOrder.push_back(name);
        }
        else if (line == "end_header")
        {
            headerDone = true;
            break;
        }
    }

    if (!headerDone)
        throw std::runtime_error("[PLYLoader] Malformed PLY: end_header not found");
    if (vertexCount <= 0)
        throw std::runtime_error("[PLYLoader] Malformed PLY: no vertices found");

    // ===== build property index map =====
    std::unordered_map<std::string, int> idx;
    for (int i = 0; i < (int)propertyOrder.size(); i++)
        idx[propertyOrder[i]] = i;

    auto require = [&](const std::string &name) {
        if (idx.find(name) == idx.end())
            throw std::runtime_error("[PLYLoader] Missing property: " + name);
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
    const int stride = (int)propertyOrder.size() * sizeof(float);
    std::vector<float> row(propertyOrder.size());

    std::vector<Gaussian3D> splats;
    splats.reserve(vertexCount);

    for (int i = 0; i < vertexCount; i++)
    {
        file.read(reinterpret_cast<char *>(row.data()), stride);
        if (!file)
            throw std::runtime_error("[PLYLoader] Unexpected end of file at vertex " + std::to_string(i));

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

        // DC SH coefficients
        g.r = row[i_dc0];
        g.g = row[i_dc1];
        g.b = row[i_dc2];

        // logit-opacity stored as-is, sigmoid applied in GaussActivLayer
        g.opacity = row[i_opacity];

        splats.push_back(g);
    }

    std::cout << "[PLYLoader] Loaded " << vertexCount << " splats from " << path << "\n";
    return splats;
}
