#include <iostream>
#include <chrono>

#include "tensor/tensor.h"
#include "nn/modules.h"
#include "nn/models.h"
#include "random/random.h"
#include "nn/gguf_reader.h"
#include "nn/tokenizer.h"

using namespace blass;

class MyModule : public nn::Module<float> {
public:
    std::shared_ptr<nn::Softmax<float>> softmax_layer;

    MyModule() {
        name = "MyModule";
        softmax_layer = std::make_shared<nn::Softmax<float>>();
        register_module("softmax", softmax_layer);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        auto output = input * 2.0f + 1.0f;
        output = (*softmax_layer)(output);
        return output;
    }
};

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
        std::cout << "Slice element " << i << ": " << slice(i) << "\n";
    }

    slice = tensor[1];
    for (size_t i = 0; i < slice.size(); ++i) {
        std::cout << "Slice element " << i << ": " << slice(i) << "\n";
    }

    std::cout << "\nView tests\n";
    std::cout << "Tensor:\n" << tensor.to_string() << "\n";
    Tensor<float> viewed = tensor.view({3, -1});
    std::cout << "Viewed Tensor:\n" << viewed.to_string() << "\n";
    Tensor<float> flat = tensor.view({-1});
    std::cout << "Flattened Tensor:\n" << flat.to_string() << "\n";
    
    std::cout << "\nTranspose tests\n";
    std::cout << "Transposed Tensor A:\n" << tensor.transpose().to_string() << "\n";
    std::cout << "Transposed Tensor B:\n" << tensor_2.transpose().to_string() << "\n";

    std::cout << "\nBroadcast tests\n";
    Tensor<float> tensor_a = {1, 2, 3};
    Tensor<float> broadcasted_a = tensor_a.broadcast({5, 3});
    std::cout << "Broadcasted Tensor:\n" << broadcasted_a.to_string() << "\n";

    Tensor<float> tensor_b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::cout << "\n";
    std::cout << "Sum of " << tensor_a.to_string() << " and " << tensor_b.to_string() << ":\n";
    std::cout << (tensor_a + tensor_b).to_string() << "\n";
    
    // matmul test
    Tensor<float> mat_a = {{{1, 2, 3}, {4, 5, 6}}, {{1, 0, 0}, {0, 1, 0}}};
    Tensor<float> mat_b = {{7, 8}, {9, 10}, {11, 12}};
    Tensor<float> mat_result = matmul(mat_a, mat_b);
    std::cout << "\nMatrix Multiplication Result:\n" << mat_result.to_string() << "\n";
    Tensor<float> mat_result_btrans = matmul(mat_a, mat_b.transpose2D(), true);
    std::cout << "\nMatrix B Transposed:\n" << mat_b.transpose2D().to_string() << "\n";
    std::cout << "\nMatrix Multiplication with B Transposed Result:\n" << mat_result_btrans.to_string() << "\n";

    // convolve1D test
    Tensor<float> input = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
    Tensor<float> kernel = {{1, 0, -1}, {0, 1, 0}};
    Tensor<float> conv_result = convolve1D(input, kernel, 0);
    std::cout << "\nConvolution Result:\n" << conv_result.to_string() << "\n";

    std::shared_ptr<MyModule> my_module = std::make_shared<MyModule>();
    Tensor<float> input_tensor = Tensor<float>::rand({4, 2});
    Tensor<float> output_tensor = (*my_module)(input_tensor);
    std::cout << "\nInput Tensor:\n" << input_tensor.to_string() << "\n";
    std::cout << "\nMyModule Output:\n" << output_tensor.to_string() << "\n";

    // set time as seed
    randomt::set_seed(std::chrono::system_clock::now().time_since_epoch().count());
    Tensor<float> rand = Tensor<float>::randn({3, 3}, 0.0f, 1.0f);
    std::cout << "\nRandom Normal Tensor:\n" << rand.to_string() << "\n";

    models::Qwen2Model qwen_model;
    qwen_model.load_model("/home/pichu2405/win/Documents/Work/Projects/test_blass/Qwen2.5-0.5B-Instruct-f16.gguf");
    std::string test = "Test abc 1234 1+1=2 aaaaaaaa 12345678910";
    auto res = qwen_model.tk.encode(test);
    Tensor<float> model_output = qwen_model.run_inference(res);

    std::cout << "\nModel output shape: ";
    for (const auto& dim : model_output.get_shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";

    return 0;
}