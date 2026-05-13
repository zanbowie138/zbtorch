#include <core/storage.h>
#include <stdexcept>

Storage make_cuda_storage(size_t) {
    throw std::runtime_error("CUDA: not implemented");
}
