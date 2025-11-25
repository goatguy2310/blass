#include <gtest/gtest.h>
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"

using namespace blass;
using namespace utils;

// Tensor<double> naive_scalar_ops(Tensor<double> t, double scalar, const char op) {
//     Tensor<double> result = Tensor<double>::from_shape(t.get_shape());
//     for (size_t i = 0; i < t.size(); ++i) {
//         result(i) = utils::scalar_op<op, double>(t(i), scalar);
//     }
//     return result;
// }

// Tensor<double> naive_tensor_ops(Tensor<double> a, Tensor<double> b, const char op) {
//     if (a.get_shape() != b.get_shape()) {
//         throw std::invalid_argument("Shapes do not match for naive tensor ops");
//     }
//     Tensor<double> result = Tensor<double>::from_shape(a.get_shape());
//     for (size_t i = 0; i < a.size(); ++i) {
//         result(i) = utils::scalar_op<op, double>(a(i), b(i));
//     }
//     return result;
// }

const double EPSILON = 1e-9;

TEST(ElemwiseTest, AddScalar) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);

    double scalar = 3.0;
    Tensor<double> a_modified = a + scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(a_modified(i, j), a(i, j) + scalar, EPSILON);
        }
    }
}

TEST(ElemwiseTest, SubtractScalar) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);

    double scalar = 3.0;
    Tensor<double> a_modified = a - scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(a_modified(i, j), a(i, j) - scalar, EPSILON);
        }
    }
}

TEST(ElemwiseTest, MultiplyScalar) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);

    double scalar = 3.0;
    Tensor<double> a_modified = a * scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(a_modified(i, j), a(i, j) * scalar, EPSILON);
        }
    }
}

TEST(ElemwiseTest, DivideScalar) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);

    double scalar = 3.0;
    Tensor<double> a_modified = a / scalar;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(a_modified(i, j), a(i, j) / scalar, EPSILON);
        }
    }
}

TEST(ElemwiseTest, AddTensorSimple) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);

    Tensor<double> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(result(i, j), a(i, j) + b(i, j), EPSILON) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(ElemwiseTest, MultiplyTensorSimple) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);

    Tensor<double> result = a * b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(result(i, j), a(i, j) * b(i, j), EPSILON) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(ElemwiseTest, AddTensorBroadcast) {
    Tensor<double> a = Tensor<double>::fill_random({100, 100}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({1, 100}, 0.0, 10.0);

    Tensor<double> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            EXPECT_NEAR(result(i, j), a(i, j) + b(0, j), EPSILON) << " at index (" << i << ", " << j << ")";
        }
    }
}

TEST(ElemwiseTest, AddTensorBroadcastLargeDim) {
    Tensor<double> a = Tensor<double>::fill_random({30, 50, 1, 20}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({1, 50, 40, 1}, 0.0, 10.0);

    Tensor<double> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            for (size_t k = 0; k < a.get_shape(2); ++k) {
                for (size_t l = 0; l < a.get_shape(3); ++l) {
                    EXPECT_NEAR(result(i, j, k, l), a(i, j, 0, l) + b(0, j, k, 0), EPSILON)
                        << " at index (" << i << ", " << j << ", " << k << ", " << l << ")";
                }
            }
        }
    }
}

TEST(ElemwiseTest, AddTensorBroadcastOffset) {
    Tensor<double> a = Tensor<double>::fill_random({40, 30, 20}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({30, 1}, 0.0, 10.0);

    Tensor<double> result = a + b;

    for (size_t i = 0; i < a.get_shape(0); ++i) {
        for (size_t j = 0; j < a.get_shape(1); ++j) {
            for (size_t k = 0; k < a.get_shape(2); ++k) {
                EXPECT_NEAR(result(i, j, k), a(i, j, k) + b(j, 0), EPSILON)
                    << " at index (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST(ElemwiseTest, AddTensorNonContiguousTranspose1) {
    Tensor<double> a = Tensor<double>::fill_random({20, 20, 20}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({20, 20, 20}, 0.0, 10.0).transpose();

    Tensor<double> result = a + b;

    for (size_t i = 0; i < result.get_shape(0); ++i) {
        for (size_t j = 0; j < result.get_shape(1); ++j) {
            for (size_t k = 0; k < result.get_shape(2); ++k) {
                EXPECT_NEAR(result(i, j, k), a(i, j, k) + b(i, j, k), EPSILON)
                    << " at index (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST(ElemwiseTest, AddTensorNonContiguousTranspose2) {
    Tensor<double> a = Tensor<double>::fill_random({50, 20, 1}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({20, 1, 50}, 0.0, 10.0).transpose();

    Tensor<double> result = a + b;

    for (size_t i = 0; i < result.get_shape(0); ++i) {
        for (size_t j = 0; j < result.get_shape(1); ++j) {
            for (size_t k = 0; k < result.get_shape(2); ++k) {
                EXPECT_NEAR(result(i, j, k), a(i, j, 0) + b(i, 0, k), EPSILON)
                    << " at index (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST(ElemwiseTest, AddTensorNonContiguousView1) {
    Tensor<double> a = Tensor<double>::fill_random({20, 20, 20}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({8000}, 0.0, 10.0).view({20, 20, 20});

    Tensor<double> result = a + b;

    for (size_t i = 0; i < result.get_shape(0); ++i) {
        for (size_t j = 0; j < result.get_shape(1); ++j) {
            for (size_t k = 0; k < result.get_shape(2); ++k) {
                EXPECT_NEAR(result(i, j, k), a(i, j, k) + b(i, j, k), EPSILON)
                    << " at index (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST(ElemwiseTest, AddTensorNonContiguousView2) {
    Tensor<double> a = Tensor<double>::fill_random({50, 1, 20}, 0.0, 10.0);
    Tensor<double> b = Tensor<double>::fill_random({400}, 0.0, 10.0).view({1, 20, 20});

    Tensor<double> result = a + b;

    for (size_t i = 0; i < result.get_shape(0); ++i) {
        for (size_t j = 0; j < result.get_shape(1); ++j) {
            for (size_t k = 0; k < result.get_shape(2); ++k) {
                EXPECT_NEAR(result(i, j, k), a(i, 0, k) + b(0, j, k), EPSILON)
                    << " at index (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}