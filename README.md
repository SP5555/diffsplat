# diffsplat
A 2D (maybe 3D as well) differentiable Gaussian splatting renderer in CUDA!

Random splats initialize on screen and optimize toward a target image, **live**.

## TODO
- [ ] Proper NDC -> pixel space transform. Don't bake in aspect ratios everywhere bruh
- [ ] Density Control to adaptively split, clone and prune splats based on gradients
- [ ] PLY file loading for feedforward 3DGS rendering
- [x] Backward pass (T_final division trick)
- [x] Adam optimizer
- [x] Watch splats converge live

---

## Dependencies
- CUDA Toolkit 11.8+
- OpenGL 3.3+
- GLFW3 (`sudo apt install libglfw3-dev`)
- GLM (`sudo apt install libglm-dev`)
- GLAD (included in `include/`)
- stb_image (included in `include/`)

## How to build
```bash
# Clone the repository
git clone https://github.com/SP5555/differentiable-splat.git
cd differentiable-splat

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run with default demo
./2dgs_cuda

# Run with custom resolution and target image
./2dgs_cuda -w 1280 -h 720 -i path/to/target/image.png # jpg works too
```

> Adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU.
> 75 = Turing, 86 = Ampere, 89 = Ada, 90 = Hopper.