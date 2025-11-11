#include <iostream>

#include "tensor/tensor.h"
using namespace blass;

int main() {
    Tensor<float> tensor = {{1, 2, 3}, {4, 5, 6}};
    Tensor<float> tensor_2 = {{7, 8}, {10, 11}, {13, 14}};
    
    Tensor<float> result = matmul_2d(tensor, tensor_2);
    for (size_t i = 0; i < result.get_shape(0); ++i) {
        for (size_t j = 0; j < result.get_shape(1); ++j) {
            std::cout << result(i, j) << " ";
        }
        std::cout << "\n";
    }
    
    tensor = tensor + tensor;

    std::cout << "First element: " << tensor.scalar() << "\n";
    std::cout << "Tensor shape: ";
    for (const auto& dim : tensor.get_shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";

    Tensor<float> slice = tensor[0];
    for (size_t i = 0; i < slice.size(); ++i) {
        std::cout << "Slice element " << i << ": " << slice[i].scalar() << "\n";
    }

    slice = tensor[1];
    for (size_t i = 0; i < slice.size(); ++i) {
        std::cout << "Slice element " << i << ": " << slice[i].scalar() << "\n";
    }

    return 0;
}