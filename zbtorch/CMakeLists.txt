cmake_minimum_required(VERSION 3.15...4.0)

project(zbtorch_ext LANGUAGES CXX)

# Find Dependencies (e.g., via vcpkg)
find_package(OpenBLAS CONFIG REQUIRED)

option(ZBTORCH_USE_CUDA "Build CUDA kernels" OFF)

if (ZBTORCH_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src/zbtorch_ext)

