cmake_minimum_required(VERSION 3.22)
project(2dgs_cuda LANGUAGES C CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architecture (adjust to your GPU)
# Common values: 75 = Turing, 86 = Ampere, 89 = Ada, 90 = Hopper
# You can also set this via: cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Dependencies
find_package(OpenGL REQUIRED)

# GLFW: prefer system install
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW not found")
endif()

# GLM: prefer system install
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(FATAL_ERROR "GLM not found")
endif()

# Sources
set(CPP_SOURCES
    src/main.cpp
    src/glad.c
    src/app/app.cpp
    src/input/input.cpp

    src/pipelines/gauss_img_fitter.cpp
    src/loaders/image_loader.cpp
)

set(CUDA_SOURCES
    src/gaussian/gaussian.cu
    src/optimizers/adam.cu
    src/utils/cuda_utils.cu
    src/layers/ndc_project_layer.cu
    src/layers/rasterize_layer.cu
    src/layers/mse_loss_layer.cu
)

# Executable
add_executable(2dgs_cuda
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

get_target_property(SOURCES 2dgs_cuda SOURCES)
message(STATUS "Sources: ${SOURCES}")

# Include paths
target_include_directories(2dgs_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(2dgs_cuda PRIVATE
    OpenGL::GL
    glfw
    glm::glm
    cuda
)

# CUDA-specific flags
set_target_properties(2dgs_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(2dgs_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")