cmake_minimum_required(VERSION 3.22)
project(diffsplat LANGUAGES C CXX CUDA)

# ===== Standards =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===== CUDA Architecture =====
# Common values: 75=Turing, 86=Ampere, 89=Ada, 90=Hopper
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 90)
endif()

# ===== Dependencies =====
find_package(OpenGL REQUIRED)

find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW not found")
endif()

find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "GLM not found via find_package, trying third_party/glm")
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/glm")
        add_library(glm::glm INTERFACE IMPORTED)
        set_target_properties(glm::glm PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/glm"
        )
    else()
        message(FATAL_ERROR "GLM not found and no third_party/glm fallback exists")
    endif()
endif()

# ===== Shared sources =====
set(CPP_SHARED_SOURCES
    src/glad.c
    src/input/input.cpp
    src/app/app_base.cpp
    src/loaders/image_saver.cpp
)

set(CUDA_SHARED_SOURCES
    src/utils/cuda_utils.cu
    src/layers/gauss_activ_layer.cu
    src/layers/ndc_project_layer.cu
    src/layers/persp_project_layer.cu
    src/layers/rasterize_layer.cu
    src/layers/mse_loss_layer.cu
    src/optimizers/adam.cu
)

# ===== Helper: apply common settings to a target =====
function(diffsplat_target_setup TARGET)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${TARGET} PRIVATE
        OpenGL::GL
        glfw
        glm::glm
        cuda
    )
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            -lineinfo
        >
    )
endfunction()

# ===== imgfitapp: differentiable image fitter =====
add_executable(imgfitapp
    src/main_imgfit.cpp
    src/app/app_imgfit.cpp
    src/loaders/image_loader.cpp
    src/pipelines/gauss_img_fitter.cpp
    ${CPP_SHARED_SOURCES}
    ${CUDA_SHARED_SOURCES}
)
diffsplat_target_setup(imgfitapp)

# ===== plyviewapp: PLY scene viewer =====
add_executable(plyviewapp
    src/main_plyview.cpp
    src/app/app_plyview.cpp
    src/loaders/ply_loader.cpp
    src/camera/camera.cpp
    src/pipelines/gauss_ply_viewer.cpp
    ${CPP_SHARED_SOURCES}
    ${CUDA_SHARED_SOURCES}
)
diffsplat_target_setup(plyviewapp)

# ===== Status =====
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")