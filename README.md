# diffsplat
A differentiable Gaussian splatting renderer built from scratch in CUDA.

Implements the full pipeline, forward rasterization, analytic backward pass, and Adam
optimization, entirely on the GPU, with no deep learning framework dependencies.

## TODO
- [ ] Density Control to adaptively split, clone and prune splats based on gradients
- [X] Build a device for 3D feedforward rendering
- [X] World space to NDC layer with proper camera transforms
- [X] PLY file loading for feedforward 3DGS rendering
- [x] Make modular base app so specific purpose apps can build on top of it
- [x] Modularize the pipeline into "layers" for PyTorch-like code
- [x] Proper NDC → pixel space transform
- [x] Watch splats converge live
- [x] Adam optimizer
- [x] Backward pass (T_final division trick)

---

## Dependencies
- CUDA Toolkit 11.8+
- OpenGL 3.3+ (provided by your GPU driver, no install needed)
- GLFW3 (`sudo apt install libglfw3-dev`)
- GLM (`sudo apt install libglm-dev`)
- GLAD (included in `include/`)
- stb_image (included in `include/`)

## Build
```sh
git clone https://github.com/SP5555/diffsplat.git
cd diffsplat
```

The easy way, use the build script.
```sh
./build.sh
```

Or manually:
```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

If you need to target a specific architecture only (e.g. for faster compile times):
```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
make -j$(nproc)
```

> Common architecture values: 75 = Turing (RTX 20xx), 86 = Ampere (RTX 30xx), 89 = Ada (RTX 40xx), 90 = Hopper, 120 = Blackwell (RTX 50xx)
> Not sure which one you have? `nvidia-smi` will tell you the GPU model. And look it up: https://developer.nvidia.com/cuda/gpus

---

## Apps

### imgfitapp (Image Fitter)
Randomly initializes a cloud of 3D Gaussians and optimizes them toward a target image using gradient descent, fully on the GPU. Watch the splats converge **live** as the renderer learns to reconstruct the image iteration by iteration.

```sh
./build/imgfitapp --width 1280 --height 720 --image path/to/image.png --splat-count 50000
```

### plyviewapp (PLY Scene Viewer)
<p align="center">
  <img src="assets/train.png" width="720"/>
  <br><em><a href="https://huggingface.co/datasets/Voxel51/gaussian_splatting">Train Scene</a></em>
</p>
Loads a pre-trained 3D Gaussian Splatting scene from a `.ply` file and renders it in real time with a free-orbit camera. Accepts any PLY file produced by standard 3DGS training pipelines.

```sh
./build/plyviewapp --scene path/to/scene.ply --scale 1.0
```

---

## Troubleshooting

### GLM not found
If CMake can't find `glm`, fix it with either:
```sh
# Option 1: install system package (recommended)
sudo apt install libglm-dev

# Option 2: pull via submodule (no sudo needed)
git submodule update --init --recursive
```

### Hybrid GPU systems (Linux, e.g. AMD iGPU + NVIDIA dGPU)
By default the renderer falls back to a host-copy display path (GPU→CPU→GPU) since
CUDA and OpenGL are on different devices. To force direct GPU display:
```sh
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/<app>
```

### Black Screen
Most likely a CUDA architecture mismatch.
If you're on an older or newer GPU, rebuild with the correct architecture manually:
```sh
rm -rf build            # nuke the broken build
mkdir build && cd build # go back in

cmake .. -DCMAKE_CUDA_ARCHITECTURES=75   # Turing (RTX 20xx)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # Ampere (RTX 30xx)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89   # Ada (RTX 40xx)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120  # Blackwell (RTX 50xx)
```
Not sure which architecture you need? Check https://developer.nvidia.com/cuda/gpus