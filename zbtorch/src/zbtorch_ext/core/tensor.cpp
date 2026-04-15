#include <cblas.h>
// class Tensor {
//
// };
#include <zbtorch_ext/tensor.h>

int add(int a, int b)
{
    return a + b;
}

/**
 * Multiplies two square matrices: C = A * B
 * @param n Dimension of the square matrices
 * @param A Pointer to matrix A (flat array)
 * @param B Pointer to matrix B (flat array)
 * @param C Pointer to result matrix C (flat array)
 */
void multiply_matrices(int n, double *A, double *B, double *C) {
    // dgemm: Double-precision General Matrix Multiplication
    // C = alpha * op(A) * op(B) + beta * C
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha, A, n, B, n, beta, C, n);
}