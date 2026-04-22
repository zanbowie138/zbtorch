cmake_minimum_required(VERSION 3.15...4.0)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(zbtorch_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)

# Find Dependencies (e.g., via vcpkg)
find_package(OpenBLAS CONFIG REQUIRED)

find_package(CUDAToolkit REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src/zbtorch_ext)

