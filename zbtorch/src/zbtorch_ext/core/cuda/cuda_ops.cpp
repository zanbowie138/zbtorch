#include <core/ops.h>
#include <stdexcept>

static void placeholder() { throw std::runtime_error("not implemented"); }

static void cuda_add   (float*, const float*, const float*, int)         { placeholder(); }
static void cuda_mul   (float*, const float*, const float*, int)         { placeholder(); }
static void cuda_sub   (float*, const float*, const float*, int)         { placeholder(); }
static void cuda_div   (float*, const float*, const float*, int)         { placeholder(); }
static void cuda_scale (float*, const float*, float, int)                { placeholder(); }
static void cuda_negate(float*, const float*, int)                       { placeholder(); }
static void cuda_pow   (float*, const float*, float, int)                { placeholder(); }
static void cuda_exp   (float*, const float*, int)                       { placeholder(); }
static void cuda_log   (float*, const float*, int)                       { placeholder(); }
static void cuda_relu  (float*, const float*, int)                       { placeholder(); }
static void cuda_tanh  (float*, const float*, int)                       { placeholder(); }
static void cuda_sigmoid(float*, const float*, int)                      { placeholder(); }

static void cuda_gemm(bool, bool, int, int, int,
                      float, const float*, int, const float*, int,
                      float, float*, int)                                { placeholder(); }

static void cuda_axpy       (int, float, const float*, float*)           { placeholder(); }
static void cuda_mul_acc    (float*, const float*, const float*, int)    { placeholder(); }
static void cuda_div_acc_lhs(float*, const float*, const float*, int)   { placeholder(); }
static void cuda_div_acc_rhs(float*, const float*, const float*, const float*, int) { placeholder(); }
static void cuda_pow_acc    (float*, const float*, const float*, float, int) { placeholder(); }
static void cuda_log_acc    (float*, const float*, const float*, int)    { placeholder(); }
static void cuda_relu_bwd   (float*, const float*, const float*, int)   { placeholder(); }
static void cuda_tanh_bwd   (float*, const float*, const float*, int)   { placeholder(); }
static void cuda_sigmoid_bwd(float*, const float*, const float*, int)   { placeholder(); }

static void cuda_zero(float*, int)        { placeholder(); }
static void cuda_fill(float*, int, float) { placeholder(); }

static const Ops cuda_ops_table = {
    cuda_add, cuda_mul, cuda_sub, cuda_div,
    cuda_scale, cuda_negate, cuda_pow,
    cuda_exp, cuda_log, cuda_relu, cuda_tanh, cuda_sigmoid,
    cuda_gemm,
    cuda_axpy, cuda_mul_acc, cuda_div_acc_lhs, cuda_div_acc_rhs,
    cuda_pow_acc, cuda_log_acc,
    cuda_relu_bwd, cuda_tanh_bwd, cuda_sigmoid_bwd,
    cuda_zero, cuda_fill
};

const Ops& get_cuda_ops() { return cuda_ops_table; }
