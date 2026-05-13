#pragma once
#include <cstddef>
#include <memory>
#include "device.h"

// Contiguous, device-aware buffer of float32.
// ptr is always the raw device pointer:
//   CPU -> host heap (new float[])
//   CUDA -> device heap (cudaMalloc)
struct Storage {
    std::shared_ptr<float> ptr;
    size_t size = 0;
    Device device = CPU;

    float* data() const { return ptr.get(); }
    bool valid() const { return ptr != nullptr; }
};

// Factories
Storage make_cpu_storage(size_t size);
Storage make_cuda_storage(size_t size);

// Dispatch to the right factory based on d.
Storage make_storage(size_t size, Device d);

// Copy a buffer, optionally moving it to a different device.
// CPU->CPU: host memcpy
Storage copy_storage(const Storage& src, Device dst);
