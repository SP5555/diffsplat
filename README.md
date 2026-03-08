# diffsplat
A 2D (maybe 3D as well) differentiable Gaussian splatting renderer in CUDA!

<!-- Random splats initialize on screen and optimize toward a target image, live.
Built mainly to learn CUDA, the 3DGS pipeline, and hand-written backward passes. -->

## TODO
- [ ] Backward pass (T_final division trick)
- [ ] Adam optimizer
- [ ] Watch splats converge live
- [ ] PLY file loading for feedforward 3DGS rendering

---

## Dependencies
- CUDA Toolkit 11.8+
- OpenGL 3.3+
- GLFW3 (`sudo apt install libglfw3-dev`)
- GLM (`sudo apt install libglm-dev`)
- GLAD (included in `include/`)

## How to build
1. Clone the repository
```bash
git clone https://github.com/SP5555/differentiable-splat.git
cd differentiable-splat
```

2. Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

3. Run
```bash
./2dgs_cuda
```

> Adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU.
> 75 = Turing, 86 = Ampere, 89 = Ada, 90 = Hopper.