#include <core/ops.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = a[i] + b[i];
}
static void cuda_add(float* out, const float* a, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    add_kernel<<<blocks, tpb>>>(out, a, b, n);
}

__global__ void mul_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = a[i] * b[i];
}
static void cuda_mul(float* out, const float* a, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    mul_kernel<<<blocks, tpb>>>(out, a, b, n);
}

__global__ void sub_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = a[i] - b[i];
}
static void cuda_sub(float* out, const float* a, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    sub_kernel<<<blocks, tpb>>>(out, a, b, n);
}

__global__ void div_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = a[i] / b[i];
}
static void cuda_div(float* out, const float* a, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    div_kernel<<<blocks, tpb>>>(out, a, b, n);
}

__global__ void scale_kernel(float* out, const float* a, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = a[i] * s;
}
static void cuda_scale(float* out, const float* a, float s, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    scale_kernel<<<blocks, tpb>>>(out, a, s, n);
}

__global__ void negate_kernel(float* out, const float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = -a[i];
}
static void cuda_negate(float* out, const float* a, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    negate_kernel<<<blocks, tpb>>>(out, a, n);
}

__global__ void pow_kernel(float* out, const float* a, float e, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = powf(a[i], e);
}
static void cuda_pow(float* out, const float* a, float e, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    pow_kernel<<<blocks, tpb>>>(out, a, e, n);
}

__global__ void exp_kernel(float* out, const float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = expf(a[i]);
}
static void cuda_exp(float* out, const float* a, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    exp_kernel<<<blocks, tpb>>>(out, a, n);
}

__global__ void log_kernel(float* out, const float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = logf(a[i]);
}
static void cuda_log(float* out, const float* a, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    log_kernel<<<blocks, tpb>>>(out, a, n);
}

__global__ void relu_kernel(float* out, const float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
static void cuda_relu(float* out, const float* a, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    relu_kernel<<<blocks, tpb>>>(out, a, n);
}

__global__ void tanh_kernel(float* out, const float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = tanhf(a[i]);
}
static void cuda_tanh(float* out, const float* a, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    tanh_kernel<<<blocks, tpb>>>(out, a, n);
}

__global__ void sigmoid_kernel(float* out, const float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = 1.0f / (1.0f + expf(-a[i]));
}
static void cuda_sigmoid(float* out, const float* a, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    sigmoid_kernel<<<blocks, tpb>>>(out, a, n);
}

// ---------------------------------------------------------------------------
// GEMM — one thread per output element, row-major
// ---------------------------------------------------------------------------

__global__ void gemm_kernel(bool transA, bool transB,
                             int M, int N, int K,
                             float alpha, const float* A, int lda,
                                          const float* B, int ldb,
                             float beta,        float* C, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float av = transA ? A[k * lda + row] : A[row * lda + k];
        float bv = transB ? B[col * ldb + k] : B[k * ldb + col];
        sum += av * bv;
    }
    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
}
static void cuda_gemm(bool transA, bool transB,
                      int M, int N, int K,
                      float alpha, const float* A, int lda,
                                   const float* B, int ldb,
                      float beta,        float* C, int ldc) {
    dim3 tpb(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<blocks, tpb>>>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ---------------------------------------------------------------------------
// Backward / gradient accumulation kernels
// ---------------------------------------------------------------------------

__global__ void axpy_kernel(float* acc, float s, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += s * src[i];
}
static void cuda_axpy(int n, float s, const float* src, float* acc) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    axpy_kernel<<<blocks, tpb>>>(acc, s, src, n);
}

__global__ void mul_acc_kernel(float* acc, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += a[i] * b[i];
}
static void cuda_mul_acc(float* acc, const float* a, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    mul_acc_kernel<<<blocks, tpb>>>(acc, a, b, n);
}

__global__ void div_acc_lhs_kernel(float* acc, const float* g, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += g[i] / b[i];
}
static void cuda_div_acc_lhs(float* acc, const float* g, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    div_acc_lhs_kernel<<<blocks, tpb>>>(acc, g, b, n);
}

__global__ void div_acc_rhs_kernel(float* acc, const float* g,
                                   const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += -(a[i] / (b[i] * b[i])) * g[i];
}
static void cuda_div_acc_rhs(float* acc, const float* g,
                              const float* a, const float* b, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    div_acc_rhs_kernel<<<blocks, tpb>>>(acc, g, a, b, n);
}

__global__ void pow_acc_kernel(float* acc, const float* x, const float* g, float e, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += e * powf(x[i], e - 1.0f) * g[i];
}
static void cuda_pow_acc(float* acc, const float* x, const float* g, float e, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    pow_acc_kernel<<<blocks, tpb>>>(acc, x, g, e, n);
}

__global__ void log_acc_kernel(float* acc, const float* x, const float* g, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += g[i] / x[i];
}
static void cuda_log_acc(float* acc, const float* x, const float* g, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    log_acc_kernel<<<blocks, tpb>>>(acc, x, g, n);
}

__global__ void relu_bwd_kernel(float* acc, const float* out, const float* g, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += (out[i] > 0.0f ? 1.0f : 0.0f) * g[i];
}
static void cuda_relu_bwd(float* acc, const float* out, const float* g, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    relu_bwd_kernel<<<blocks, tpb>>>(acc, out, g, n);
}

__global__ void tanh_bwd_kernel(float* acc, const float* out, const float* g, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += (1.0f - out[i] * out[i]) * g[i];
}
static void cuda_tanh_bwd(float* acc, const float* out, const float* g, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    tanh_bwd_kernel<<<blocks, tpb>>>(acc, out, g, n);
}

__global__ void sigmoid_bwd_kernel(float* acc, const float* out, const float* g, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) acc[i] += out[i] * (1.0f - out[i]) * g[i];
}
static void cuda_sigmoid_bwd(float* acc, const float* out, const float* g, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    sigmoid_bwd_kernel<<<blocks, tpb>>>(acc, out, g, n);
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

__global__ void zero_kernel(float* buf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) buf[i] = 0.0f;
}
static void cuda_zero(float* buf, int n) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    zero_kernel<<<blocks, tpb>>>(buf, n);
}

__global__ void fill_kernel(float* buf, int n, float v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) buf[i] = v;
}
static void cuda_fill(float* buf, int n, float v) {
    int tpb = 256, blocks = (n + tpb - 1) / tpb;
    fill_kernel<<<blocks, tpb>>>(buf, n, v);
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

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
