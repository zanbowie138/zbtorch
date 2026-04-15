#include <zbtorch_ext/tensor.h>
#include <cstdio>
#include <iostream>
#include <memory>

int main() {
    auto a = std::make_shared<Tensor>(std::vector<float>{1, 2, 3,
        4, 5, 6,
        7, 8, 9}, std::vector<size_t>{3, 3});

    auto b = std::make_shared<Tensor>(std::vector<float>{
        9, 8, 7,
        6, 5, 4,
        3, 2, 1}, std::vector<size_t>{3, 3});

    printf("Executing Matrix Multiplication using Tensor class...\n");

    auto c = std::make_shared<Tensor>(a->matmul(*b));

    std::cout << "Result Matrix C:\n" << *c << std::endl;

    return 0;
}
