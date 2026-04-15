# Developer Setup

Hybrid C++/Python ML framework. Uses **vcpkg** for C++ deps (OpenBLAS), **uv** for Python packaging, and **scikit-build-core** + **CMake** as the build system.

## Prerequisites

- CMake >= 3.15
- C++ compiler (MSVC on Windows; GCC/Clang on Linux/macOS)
- [vcpkg](https://vcpkg.io)
- [uv](https://docs.astral.sh/uv/)

## 1. Install C++ Dependencies

Use vcpkg to install OpenBLAS. Adjust the triplet for your platform:

| Platform    | Triplet       |
|-------------|---------------|
| Windows x64 | `x64-windows` |
| Linux x64   | `x64-linux`   |
| macOS x64   | `x64-osx`     |
| macOS arm64 | `arm64-osx`   |

```sh
vcpkg install openblas:<triplet>
```

## 2. Set Up Python Environment

```sh
uv sync --dev
```

Creates `.venv/` and installs all dependencies, including `pybind11` (required by CMake at configure time).

## 3. Configure CMake

The CMake root is `zbtorch/`. Point CMake to vcpkg and the local `.venv`:

> **Windows:** Use forward slashes `/` in all paths — backslashes will cause failures.

**Windows**
```sh
cmake -S zbtorch -B build \
  -DCMAKE_TOOLCHAIN_FILE="<vcpkg_root>/scripts/buildsystems/vcpkg.cmake" \
  -Dpybind11_DIR="<project_root>/.venv/Lib/site-packages/pybind11/share/cmake/pybind11" \
  -DPython_EXECUTABLE="<project_root>/.venv/Scripts/python.exe"
```

**Linux / macOS**
```sh
cmake -S zbtorch -B build \
  -DCMAKE_TOOLCHAIN_FILE="<vcpkg_root>/scripts/buildsystems/vcpkg.cmake" \
  -Dpybind11_DIR="<project_root>/.venv/lib/python3.14/site-packages/pybind11/share/cmake/pybind11" \
  -DPython_EXECUTABLE="<project_root>/.venv/bin/python"
```

| Flag | Purpose |
|------|---------|
| `CMAKE_TOOLCHAIN_FILE` | Lets CMake find vcpkg packages (OpenBLAS) |
| `pybind11_DIR` | Points CMake to pybind11 config inside `.venv` |
| `Python_EXECUTABLE` | Forces CMake to use the venv interpreter |

Optional: add `-DZBTORCH_USE_CUDA=ON` to build CUDA kernels (requires CUDA Toolkit).

## 4. Build

```sh
cmake --build build
```

## 5. Build Targets

| Target               | Type           | Description |
|----------------------|----------------|-------------|
| `zbtorch_standalone` | C++ executable | Runs `test/test.cpp`; use for debugging C++ logic without Python overhead |
| `_C`                 | Shared library | pybind11 extension (`.pyd` on Windows, `.so` on Linux/macOS); cannot be run directly |

## 6. IDE Notes (CLion / VS Code)

- Set the CMake source root to `zbtorch/`, not the repo root.
- Add the three CMake flags above to your IDE's CMake profile.
- Use `zbtorch_standalone` as the run/debug target for C++ development.
