#include <core/storage.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <format>
#include <stdexcept>

Storage make_storage(size_t n, Device d) {
    switch (d) {
        case CPU:  return make_cpu_storage(n);
        case CUDA: return make_cuda_storage(n);
        default:   throw std::runtime_error("make_storage: unknown device");
    }
}

Storage copy_storage(const Storage& src, Device dst) {
    if (!src.valid())
        return {};
    if (src.device == CPU && dst == CPU) {
        Storage out = make_cpu_storage(src.size);
        std::copy(src.data(), src.data() + src.size, out.data());
        return out;
    }
    if (src.device == CUDA && dst == CUDA) {
        Storage out = make_cuda_storage(src.size);
        cudaMemcpy(out.data(), src.data(), src.size * sizeof(float), cudaMemcpyDeviceToDevice);
        return out;
    }
    if (src.device == CPU && dst == CUDA) {
        Storage out = make_cuda_storage(src.size);
        cudaMemcpy(out.data(), src.data(), src.size * sizeof(float), cudaMemcpyHostToDevice);
        return out;
    }
    if (src.device == CUDA && dst == CPU) {
        Storage out = make_cpu_storage(src.size);
        cudaMemcpy(out.data(), src.data(), src.size * sizeof(float), cudaMemcpyDeviceToHost);
        return out;
    }
    throw std::runtime_error(std::format("copy_storage: cross-device copy between {} and {} not implemented", getDeviceName(src.device), getDeviceName(dst)));
}
