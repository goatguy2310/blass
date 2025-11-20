#include <gtest/gtest.h>
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"

using namespace blass;
using namespace utils;

// Tensor<float> naive_scalar_ops(Tensor<float> t, float scalar, const char op) {
//     Tensor<float> result = Tensor<float>::from_shape(t.get_shape());
//     for (size_t i = 0; i < t.size(); ++i) {
//         result(i) = utils::scalar_op<op, float>(t(i), scalar);
//     }
//     return result;
// }

// Tensor<float> naive_tensor_ops(Tensor<float> a, Tensor<float> b, const char op) {
//     if (a.get_shape() != b.get_shape()) {
//         throw std::invalid_argument("Shapes do not match for naive tensor ops");
//     }
//     Tensor<float> result = Tensor<float>::from_shape(a.get_shape());
//     for (size_t i = 0; i < a.size(); ++i) {
//         result(i) = utils::scalar_op<op, float>(a(i), b(i));
//     }
//     return result;
// }

TEST(ElemwiseTest, AddScalar) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);

    float scalar = 3.0f;
    Tensor<float> a_modified = a + scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(a_modified(i, j), a(i, j) + scalar);
        }
    }
}

TEST(ElemwiseTest, SubtractScalar) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);

    float scalar = 3.0f;
    Tensor<float> a_modified = a - scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(a_modified(i, j), a(i, j) - scalar);
        }
    }
}

TEST(ElemwiseTest, MultiplyScalar) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);

    float scalar = 3.0f;
    Tensor<float> a_modified = a * scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(a_modified(i, j), a(i, j) * scalar);
        }
    }
}

TEST(ElemwiseTest, DivideScalar) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);

    float scalar = 3.0f;
    Tensor<float> a_modified = a / scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(a_modified(i, j), a(i, j) / scalar);
        }
    }
}

TEST(ElemwiseTest, AddTensorSimple) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);

    Tensor<float> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(result(i, j), a(i, j) + b(i, j)) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(ElemwiseTest, MultiplyTensorSimple) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);

    Tensor<float> result = a * b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(result(i, j), a(i, j) * b(i, j)) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(ElemwiseTest, AddTensorBroadcast) {
    Tensor<float> a = Tensor<float>::fill_random({1000, 1000}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({1, 1000}, 0.0f, 10.0f);

    Tensor<float> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_FLOAT_EQ(result(i, j), a(i, j) + b(0, j)) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(ElemwiseTest, AddTensorBroadcastLargeDim) {
    Tensor<float> a = Tensor<float>::fill_random({300, 50, 1, 200}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({1, 50, 400, 1}, 0.0f, 10.0f);

    Tensor<float> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            for (size_t k = 0; k < a.get_shape(2); ++k) {
                for (size_t l = 0; l < a.get_shape(3); ++l) {
                    EXPECT_FLOAT_EQ(result(i, j, k, l), a(i, j, 0, l) + b(0, j, k, 0))
                        << " at index (" << i << ", " << j << ", " << k << ", " << l << ")";
                }
            }
        }
    }
}

TEST(ElemwiseTest, AddTensorBroadcastOffset) {
    Tensor<float> a = Tensor<float>::fill_random({40, 30, 20}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({30, 1}, 0.0f, 10.0f);

    Tensor<float> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            for (size_t k = 0; k < a.get_shape(2); ++k) {
                EXPECT_FLOAT_EQ(result(i, j, k), a(i, j, k) + b(j, 0))
                    << " at index (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}