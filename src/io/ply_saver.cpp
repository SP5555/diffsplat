#include "ply_saver.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include "../utils/logs.h"

void PLYSaver::save(const std::string &path, const std::vector<Gaussian3D> &splats,
                    int sh_num_bands)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
        log_fatal("PLYSaver", "Failed to open for writing: " + path);

    int n = (int)splats.size();

    // ===== write header =====
    // property count: x y z  f_dc(3)  f_rest(3*K)  opacity  scale(3)  rot(4)
    int n_rest = sh_num_bands * 3;

    auto prop = [&](const std::string &name) {
        file << "property float " << name << "\n";
    };

    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << n << "\n";
    prop("x"); prop("y"); prop("z");
    prop("f_dc_0"); prop("f_dc_1"); prop("f_dc_2");
    for (int i = 0; i < n_rest; i++)
        prop("f_rest_" + std::to_string(i));
    prop("opacity");
    prop("scale_0"); prop("scale_1"); prop("scale_2");
    prop("rot_0"); prop("rot_1"); prop("rot_2"); prop("rot_3");
    file << "end_header\n";

    // ===== write binary data =====
    for (const auto &g : splats)
    {
        // Convert OpenGL (Y up, Z backward) back to OpenCV (Y down, Z forward)
        // by flipping Y and Z. Quaternion: negate rot_y and rot_z
        // (rot_w double-negation from Y and Z flips cancels out).
        float pos_x =  g.pos_x;
        float pos_y = -g.pos_y;
        float pos_z = -g.pos_z;
        float rot_w =  g.rot_w;
        float rot_x =  g.rot_x;
        float rot_y = -g.rot_y;
        float rot_z = -g.rot_z;

        auto wf = [&](float v) { file.write(reinterpret_cast<const char *>(&v), sizeof(float)); };

        wf(pos_x); wf(pos_y); wf(pos_z);
        wf(g.sh_dc_r); wf(g.sh_dc_g); wf(g.sh_dc_b);
        for (int i = 0; i < n_rest; i++) wf(g.sh_rest[i]);
        wf(g.logit_opacity);
        wf(g.scale_x); wf(g.scale_y); wf(g.scale_z);
        // PLY convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z
        wf(rot_w); wf(rot_x); wf(rot_y); wf(rot_z);
    }

    log_info("PLYSaver", "Saved " + std::to_string(n) + " splats to " + path);
}
