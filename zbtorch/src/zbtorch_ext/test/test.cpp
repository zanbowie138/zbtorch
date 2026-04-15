#include <zbtorch_ext/tensor.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int n = 3; // 3x3 matrices

    // Initialize matrices in row-major order
    double A[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    double B[] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };

    double *C = (double *)malloc(n * n * sizeof(double));

    printf("Executing Matrix Multiplication using OpenBLAS...\n");

    // Call the tester function
    multiply_matrices(n, A, B, C);

    // Print result
    printf("Result Matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%7.1f ", C[i * n + j]);
        }
        printf("\n");
    }

    free(C);
    return 0;
}

