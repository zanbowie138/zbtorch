#include <cuda_runtime.h>
#include <core/storage.h>
#include <stdexcept>

Storage make_cuda_storage(size_t size) {
    float* data;
    cudaMalloc(&data, size * sizeof(float));
    return {
    std::shared_ptr<float>(data, cudaFree),
    size,
    CUDA
    };
}
