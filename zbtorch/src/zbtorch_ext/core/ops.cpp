#include <core/ops.h>
#include <stdexcept>

const Ops& get_cpu_ops();
const Ops& get_cuda_ops();

const Ops& get_ops(Device d) {
    switch (d) {
        case CPU:  return get_cpu_ops();
        case CUDA: return get_cuda_ops();
        default:   throw std::runtime_error("get_ops: unknown device");
    }
}
