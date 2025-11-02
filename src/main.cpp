#include <iostream>

#include "tensor/tensor.h"

int main() {
    Tensor<float> tensor({2, 3});
    tensor = 6.7f;

    std::cout << tensor[0][0].size() << "\n";

    std::cout << "First element: " << tensor.scalar() << "\n";
    std::cout << "Tensor size: " << tensor.size() << "\n";

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