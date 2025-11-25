#include <gtest/gtest.h>
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"

using namespace blass;
using namespace utils;

const double EPSILON = 1e-9;

TEST(MatMulTest, MatMul2DSimple) {
    Tensor<double> a = Tensor<double>::fill_random({50, 30}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({30, 40}, 0.0, 10.0);

    Tensor<double> result = matmul(a, b);

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < b.get_shape(1); ++j) {
            double expected_value = 0.0;
            for (size_t k = 0; k < a.get_shape(1); ++k) {
                expected_value += a(i, k) * b(k, j);
            }
            EXPECT_NEAR(result(i, j), expected_value, EPSILON)
                << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(MatMulTest, MatMulBatched) {
    Tensor<double> a = Tensor<double>::fill_random({10, 20, 30}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({10, 30, 40}, 0.0, 10.0);

    Tensor<double> result = matmul(a, b);

    for (size_t batch = 0; batch < a.get_shape(0); ++batch) {
        for (size_t i = 0; i < a.get_shape(1); ++i) {
            for (size_t j = 0; j < b.get_shape(2); ++j) {
                double expected_value = 0.0;
                for (size_t k = 0; k < a.get_shape(2); ++k) {
                    expected_value += a(batch, i, k) * b(batch, k, j);
                }
                EXPECT_NEAR(result(batch, i, j), expected_value, EPSILON)
                    << " at index (" << batch << ", " << i << ", " << j << ")";
            }
        }
    }
}

TEST(MatMulTest, MatMulBatchedBroadcasted) {
    Tensor<double> a = Tensor<double>::fill_random({10, 20, 30}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({1, 30, 40}, 0.0, 10.0);

    Tensor<double> result = matmul(a, b);

    for (size_t batch = 0; batch < a.get_shape(0); ++batch) {
        for (size_t i = 0; i < a.get_shape(1); ++i) {
            for (size_t j = 0; j < b.get_shape(2); ++j) {
                double expected_value = 0.0;
                for (size_t k = 0; k < a.get_shape(2); ++k) {
                    expected_value += a(batch, i, k) * b(0, k, j);
                }
                EXPECT_NEAR(result(batch, i, j), expected_value, EPSILON)
                    << " at index (" << batch << ", " << i << ", " << j << ")";
            }
        }
    }
}