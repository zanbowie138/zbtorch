#pragma once
#include "device.h"

// Per-device function pointer table.
// All pointers operate on raw float* device pointers.
// Callers are responsible for ensuring all pointers belong to the same device.
struct Ops {
    // --- Element-wise forward kernels ---
    void (*add)    (float* out, const float* a, const float* b, int n);
    void (*mul)    (float* out, const float* a, const float* b, int n);
    void (*sub)    (float* out, const float* a, const float* b, int n);
    void (*div_op) (float* out, const float* a, const float* b, int n);
    void (*scale)  (float* out, const float* a, float s, int n);   // out = a * s
    void (*negate) (float* out, const float* a, int n);
    void (*pow_op) (float* out, const float* a, float e, int n);

    // --- Activation forward kernels ---
    void (*exp_op)     (float* out, const float* a, int n);
    void (*log_op)     (float* out, const float* a, int n);
    void (*relu_op)    (float* out, const float* a, int n);
    void (*tanh_op)    (float* out, const float* a, int n);
    void (*sigmoid_op) (float* out, const float* a, int n);

    // --- General matrix multiply (row-major) ---
    // C(M,N) = alpha * op(A) * op(B) + beta * C(M,N)
    // transA/transB: whether to logically transpose A or B.
    // lda, ldb, ldc: leading dimensions of A, B, C as stored in memory.
    void (*gemm)(bool transA, bool transB,
                 int M, int N, int K,
                 float alpha, const float* A, int lda,
                              const float* B, int ldb,
                 float beta,        float* C, int ldc);

    // --- Gradient accumulation kernels (backward) ---
    void (*axpy)        (int n, float s, const float* src, float* acc);         // acc += s * src
    void (*mul_acc)     (float* acc, const float* a, const float* b, int n);    // acc += a * b (elem)
    void (*div_acc_lhs) (float* acc, const float* g, const float* b, int n);   // acc += g / b
    void (*div_acc_rhs) (float* acc, const float* g,                           // acc += -a/b^2 * g
                         const float* a, const float* b, int n);
    void (*pow_acc)     (float* acc, const float* x,                           // acc += e*x^(e-1)*g
                         const float* g, float e, int n);
    void (*log_acc)     (float* acc, const float* x, const float* g, int n);   // acc += g / x
    void (*relu_bwd)    (float* acc, const float* out, const float* g, int n); // acc += (out>0)*g
    void (*tanh_bwd)    (float* acc, const float* out, const float* g, int n); // acc += (1-out^2)*g
    void (*sigmoid_bwd) (float* acc, const float* out, const float* g, int n); // acc += out*(1-out)*g

    // --- Utility ---
    void (*zero) (float* buf, int n);          // buf[i] = 0
    void (*fill) (float* buf, int n, float v); // buf[i] = v
};

// Returns the static Ops table for the given device.
// Thread-safe: tables are const after program initialization.
const Ops& get_ops(Device d);
