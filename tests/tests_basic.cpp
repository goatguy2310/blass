#include <gtest/gtest.h>
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"

using namespace blass;
using namespace utils;

float get_value(const std::shared_ptr<float[]> data, size_t i, size_t j, size_t row) {
    return data[i * row + j];
}

TEST(Basic, Access) {
    size_t col = 100, row = 100;
    std::shared_ptr<float[]> data(new float[col * row]);
    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            data[i * row + j] = static_cast<float>((i * row + j) * 2.0);
        }
    }
    Tensor<float> a = Tensor<float>(data, {col, row});

    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            EXPECT_FLOAT_EQ(a(i, j), get_value(data, i, j, row)) << " at index (" << i << ", " << j << ") using operator()";
            EXPECT_TRUE(a[i][j].is_scalar()) << " at index (" << i << ", " << j << ") using operator[] is not scalar";
            EXPECT_FLOAT_EQ(a[i][j].scalar(), get_value(data, i, j, row)) << " at index (" << i << ", " << j << ") using operator[]";
            EXPECT_TRUE(a.at({i, j}).is_scalar()) << " at index (" << i << ", " << j << ") using at() is not scalar";
            EXPECT_FLOAT_EQ(a.at({i, j}).scalar(), get_value(data, i, j, row)) << " at index (" << i << ", " << j << ") using at()";
        }
    }
}

TEST(Basic, Transpose) {
    size_t col = 100, row = 100;
    std::shared_ptr<float[]> data(new float[col * row]);
    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            data[i * row + j] = static_cast<float>((i * row + j) * 2.0);
        }
    }
    Tensor<float> a = Tensor<float>(data, {col, row}).transpose();

    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            EXPECT_FLOAT_EQ(a(i, j), get_value(data, j, i, row)) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(Basic, View) {
    size_t col = 100, row = 100;
    std::shared_ptr<float[]> data(new float[col * row]);
    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            data[i * row + j] = static_cast<float>((i * row + j) * 2.0);
        }
    }
    size_t dim1 = 50, dim2 = 20, dim3 = 10;
    Tensor<float> a = Tensor<float>(data, {col, row}).view({(int) dim1, (int) dim2, (int) dim3});

    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            size_t idx = i * row + j;
            size_t i1 = idx / (dim2 * dim3);
            size_t i2 = (idx / dim3) % dim2;
            size_t i3 = idx % dim3;
            EXPECT_FLOAT_EQ(a(i1, i2, i3), get_value(data, i, j, row)) << " at index (" << i1 << ", " << i2 << ", " << i3 << ")";
        }
    }
    
    Tensor<float> b = a.view({(int) col, (int) row});
    for (size_t i = 0; i < col; ++i) {
        for (size_t j = 0; j < row; ++j) {
            EXPECT_FLOAT_EQ(b(i, j), get_value(data, i, j, row)) << " at index (" << i << ", " << j << ") when viewed back";
        }
    }
}