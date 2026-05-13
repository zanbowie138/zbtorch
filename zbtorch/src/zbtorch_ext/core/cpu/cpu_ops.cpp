#include <core/ops.h>
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

static void cpu_add(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void cpu_mul(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
static void cpu_sub(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
}
static void cpu_div(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] / b[i];
}
static void cpu_scale(float* out, const float* a, float s, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * s;
}
static void cpu_negate(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) out[i] = -a[i];
}
static void cpu_pow(float* out, const float* a, float e, int n) {
    for (int i = 0; i < n; i++) out[i] = std::pow(a[i], e);
}
static void cpu_exp(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) out[i] = std::exp(a[i]);
}
static void cpu_log(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) out[i] = std::log(a[i]);
}
static void cpu_relu(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
static void cpu_tanh(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) out[i] = std::tanh(a[i]);
}
static void cpu_sigmoid(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) out[i] = 1.0f / (1.0f + std::exp(-a[i]));
}

static void cpu_gemm(bool transA, bool transB,
                     int M, int N, int K,
                     float alpha, const float* A, int lda,
                                  const float* B, int ldb,
                     float beta,        float* C, int ldc) {
    cblas_sgemm(CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ---------------------------------------------------------------------------
// Backward / gradient accumulation kernels
// ---------------------------------------------------------------------------

static void cpu_axpy(int n, float s, const float* src, float* acc) {
    cblas_saxpy(n, s, src, 1, acc, 1);
}
static void cpu_mul_acc(float* acc, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) acc[i] += a[i] * b[i];
}
static void cpu_div_acc_lhs(float* acc, const float* g, const float* b, int n) {
    for (int i = 0; i < n; i++) acc[i] += g[i] / b[i];
}
static void cpu_div_acc_rhs(float* acc, const float* g,
                             const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) acc[i] += -(a[i] / (b[i] * b[i])) * g[i];
}
static void cpu_pow_acc(float* acc, const float* x, const float* g, float e, int n) {
    for (int i = 0; i < n; i++) acc[i] += e * std::pow(x[i], e - 1.0f) * g[i];
}
static void cpu_log_acc(float* acc, const float* x, const float* g, int n) {
    for (int i = 0; i < n; i++) acc[i] += g[i] / x[i];
}
static void cpu_relu_bwd(float* acc, const float* out, const float* g, int n) {
    for (int i = 0; i < n; i++) acc[i] += (out[i] > 0.0f ? 1.0f : 0.0f) * g[i];
}
static void cpu_tanh_bwd(float* acc, const float* out, const float* g, int n) {
    for (int i = 0; i < n; i++) acc[i] += (1.0f - out[i] * out[i]) * g[i];
}
static void cpu_sigmoid_bwd(float* acc, const float* out, const float* g, int n) {
    for (int i = 0; i < n; i++) acc[i] += out[i] * (1.0f - out[i]) * g[i];
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

static void cpu_zero(float* buf, int n) {
    std::fill(buf, buf + n, 0.0f);
}
static void cpu_fill(float* buf, int n, float v) {
    std::fill(buf, buf + n, v);
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

static const Ops cpu_ops_table = {
    cpu_add, cpu_mul, cpu_sub, cpu_div,
    cpu_scale, cpu_negate, cpu_pow,
    cpu_exp, cpu_log, cpu_relu, cpu_tanh, cpu_sigmoid,
    cpu_gemm,
    cpu_axpy, cpu_mul_acc, cpu_div_acc_lhs, cpu_div_acc_rhs,
    cpu_pow_acc, cpu_log_acc,
    cpu_relu_bwd, cpu_tanh_bwd, cpu_sigmoid_bwd,
    cpu_zero, cpu_fill
};

const Ops& get_cpu_ops() { return cpu_ops_table; }
