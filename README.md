# diffsplat
A 2D (maybe 3D as well) differentiable Gaussian splatting renderer in CUDA!

Random splats initialize on screen and optimize toward a target image, **live**.

## TODO
- [ ] Build a device for 3D feedforward rendering
- [ ] Density Control to adaptively split, clone and prune splats based on gradients
- [ ] PLY file loading for feedforward 3DGS rendering
- [x] (**URGENT**) Modularize the pipeline into "layers" for pytorch like code.
- [x] Proper NDC -> pixel space transform. Don't bake in aspect ratios everywhere bruh
- [x] Backward pass (T_final division trick)
- [x] Adam optimizer
- [x] Watch splats converge live

---

## Dependencies
- CUDA Toolkit 11.8+
- OpenGL 3.3+ (provided by your GPU driver, no install needed)
- GLFW3 (`sudo apt install libglfw3-dev`)
- GLM (`sudo apt install libglm-dev`)
- GLAD (included in `include/`)
- stb_image (included in `include/`)

## Build
```bash
git clone https://github.com/SP5555/diffsplat.git
cd diffsplat
```

The easy way, use the build script which auto-detects nvcc and defaults to `sm_86`:
```bash
./build.sh               # defaults to Ampere (sm_86)
CUDA_ARCH=89 ./build.sh  # override for Ada
```

Or manually:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)
```

> Common architecture values: 75 = Turing, 86 = Ampere, 89 = Ada, 90 = Hopper.
> Not sure which one you have? `nvidia-smi` will tell you the GPU model. And go to: https://developer.nvidia.com/cuda/gpus

## Run
```bash
./build/2dgs_cuda                                      # default demo
./build/2dgs_cuda -w 1280 -h 720 -i path/to/image.png  # custom resolution and target
```

> jpg and png are both supported.

## Troubleshooting

### GLM not found
If CMake can't find `glm`, fix it with either:
```bash
# Option 1: install system package (recommended)
sudo apt install libglm-dev

# Option 2: pull via submodule (useful without sudo access)
git submodule update --init --recursive
```

### Hybrid GPU systems (Linux, e.g. AMD iGPU + NVIDIA dGPU)
By default the renderer falls back to a slower display path. To force the fast path:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/2dgs_cuda
```