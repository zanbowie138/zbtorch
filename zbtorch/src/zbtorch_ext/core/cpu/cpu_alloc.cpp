#include <core/storage.h>

Storage make_cpu_storage(size_t n) {
    return {
        std::shared_ptr<float>(new float[n], [](float* p){ delete[] p; }),
        n,
        CPU
    };
}
