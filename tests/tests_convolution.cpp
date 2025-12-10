#include <gtest/gtest.h>
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"

using namespace blass;
using namespace utils;

double get_value(const std::shared_ptr<double[]> data, size_t i, size_t j, size_t row) {
    return data[i * row + j];
}

TEST(ConvolutionTest, Conv1DSimpleNoPadding) {
    size_t batch_size = 10, in_channels = 50, input_length = 100, kernel_size = 7;
    Tensor<double> input = Tensor<double>::fill_random({batch_size, in_channels, input_length}, 0.0, 10.0);
    Tensor<double> kernel = Tensor<double>::fill_random({1, in_channels, kernel_size}, 0.0, 10.0);
    Tensor<double> output = convolve1D(input, kernel, false);
    Tensor<double> expected = Tensor<double>::from_shape({batch_size, 1, input_length - kernel_size + 1});
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ol = 0; ol < input_length - kernel_size + 1; ol++) {
            double sum = 0.0;
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t k = 0; k < kernel_size; k++) {
                    sum += input(b, ic, ol + k) * kernel(0, ic, k);
                }
            }
            expected(b, 0, ol) = sum;
        }
    }
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ol = 0; ol < input_length - kernel_size + 1; ol++) {
            EXPECT_NEAR(output(b, 0, ol), expected(b, 0, ol), 1e-9)
                << " at index (" << b << ", " << ol << ")";
        }
    }
}

TEST(ConvolutionTest, Conv1DSimplePadding) {
    size_t batch_size = 10, in_channels = 50, input_length = 100, kernel_size = 7;
    Tensor<double> input = Tensor<double>::fill_random({batch_size, in_channels, input_length}, 0.0, 10.0);
    Tensor<double> kernel = Tensor<double>::fill_random({1, in_channels, kernel_size}, 0.0, 10.0);
    Tensor<double> output = convolve1D(input, kernel, true);
    Tensor<double> expected = Tensor<double>::from_shape({batch_size, 1, input_length});
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ol = 0; ol < input_length; ol++) {
            double sum = 0.0;
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t k = 0; k < kernel_size; k++) {
                    int input_idx = (int)ol + (int)k - (int)(kernel_size / 2);
                    if (input_idx < 0 || input_idx >= (int)input_length)
                        continue;
                    sum += input(b, ic, input_idx) * kernel(0, ic, k);
                }
            }
            expected(b, 0, ol) = sum;
        }
    }
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ol = 0; ol < input_length; ol++) {
            EXPECT_NEAR(output(b, 0, ol), expected(b, 0, ol), 1e-9)
                << " at index (" << b << ", " << ol << ")";
        }
    }
}