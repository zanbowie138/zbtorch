#include <core/storage.h>
#include <algorithm>
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
    throw std::runtime_error("copy_storage: cross-device copy not implemented");
}
