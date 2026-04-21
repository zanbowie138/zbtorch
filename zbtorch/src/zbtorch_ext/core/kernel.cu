#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// gpu kernel
__global__ void add(int n, float *x, float *y) {
    // Calculate global thread ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to handle arrays larger than the thread count
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

// host
void runCudaTest() {
    int N = 1 << 20; // 1 Million elements (approx 4MB per array)
    float *x, *y;

    // 1. Allocate Unified Memory (accessible from CPU and GPU)
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // 2. Initialize the arrays on the CPU
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 3. Launch the kernel on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    std::cout << "Launching GPU kernel with " << numBlocks
              << " blocks of " << blockSize << " threads..." << std::endl;

    add<<<numBlocks, blockSize>>>(N, x, y);

    // 4. Wait for GPU to finish before the CPU accesses the results
    cudaDeviceSynchronize();

    // 5. Check for errors on the CPU
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = std::fmax(maxError, std::fabs(y[i] - 3.0f));
    }

    if (maxError == 0.0f) {
        std::cout << "SUCCESS! 1 million additions performed on the GPU." << std::endl;
    } else {
        std::cout << "FAILED! Max error: " << maxError << std::endl;
    }

    // 6. Free the memory
    cudaFree(x);
    cudaFree(y);
}