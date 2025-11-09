#include <iostream>

#include "tensor/tensor.h"
using namespace blass;

int main() {
    Tensor<float> tensor = {{1, 2, 3}, {4, 5, 6}};
    tensor = tensor + tensor - 2;
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