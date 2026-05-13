add_library(zbtorch_core STATIC
    core/storage.cpp
    core/tensor.cpp
    core/neuron.cpp
    core/ops.cpp
    core/cpu/cpu_alloc.cpp
    core/cpu/cpu_ops.cpp
    core/cuda/cuda_alloc.cpp
    core/cuda/cuda_ops.cu
)
target_include_directories(zbtorch_core PUBLIC include)
target_link_libraries(zbtorch_core PUBLIC
    OpenBLAS::OpenBLAS
    CUDA::cudart
)

# Python Module
pybind11_add_module(_C bindings.cpp)
target_include_directories(_C PUBLIC include)
target_link_libraries(_C PRIVATE
    zbtorch_core
    pybind11::module
    CUDA::cudart
)

# Standalone Testing Module
add_executable(zbtorch_standalone test/test.cpp)
target_link_libraries(zbtorch_standalone PRIVATE
    zbtorch_core
)

install(TARGETS _C
    LIBRARY DESTINATION zbtorch
    RUNTIME DESTINATION zbtorch   # Windows .pyd lands here
)

# On Windows, copy DLL dependencies (e.g. openblas.dll) alongside the extension
install(FILES $<TARGET_RUNTIME_DLLS:_C> DESTINATION zbtorch)
