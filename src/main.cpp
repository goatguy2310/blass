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

    std::cout << "View tests\n";
    std::cout << "Tensor:\n" << tensor.to_string() << "\n";
    Tensor<float> viewed = tensor.view({3, -1});
    std::cout << "Viewed Tensor:\n" << viewed.to_string() << "\n";
    Tensor<float> flat = tensor.view({-1});
    std::cout << "Flattened Tensor:\n" << flat.to_string() << "\n";
    
    std::cout << "Transpose tests\n";
    std::cout << "Transposed Tensor A:\n" << tensor.transpose().to_string() << "\n";
    std::cout << "Transposed Tensor B:\n" << tensor_2.transpose().to_string() << "\n";

    return 0;
}