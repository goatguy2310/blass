#include <iostream>

#include "tensor/tensor.h"

int main() {
    Tensor<float> tensor({2, 3});
    tensor[0][0] = 1.0f;
    tensor[0][1] = 2.0f;
    tensor[0][2] = 3.0f;
    tensor[1][0] = 4.0f;
    tensor[1][1] = 5.0f;
    tensor[1][2] = 6.0f;

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