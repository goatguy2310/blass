#include <cassert>

namespace blass {
    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for addition");
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data[i] = a.data[i] + b.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for subtraction");
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data[i] = a.data[i] - b.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for multiplication");
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data[i] = a.data[i] * b.data[i];
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> Tensor<T>::operator+(const Tensor<T>& b) const {
        return add(*this, b);
    }

    template <typename T>
    inline Tensor<T> Tensor<T>::operator-(const Tensor<T>& b) const {
        return subtract(*this, b);
    }

    template <typename T>
    inline Tensor<T> Tensor<T>::operator*(const Tensor<T>& b) const {
        return multiply(*this, b);
    }

    template <typename T>
    inline Tensor<T> Tensor<T>::operator+(const T& scalar) const {
        Tensor<T> result = Tensor<T>::from_shape(this->get_shape());
        for (size_t i = 0; i < this->size(); ++i) {
            result.data[i] = this->data[i] + scalar;
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> Tensor<T>::operator-(const T& scalar) const {
        Tensor<T> result = Tensor<T>::from_shape(this->get_shape());
        for (size_t i = 0; i < this->size(); ++i) {
            result.data[i] = this->data[i] - scalar;
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> Tensor<T>::operator*(const T& scalar) const {
        Tensor<T> result = Tensor<T>::from_shape(this->get_shape());
        for (size_t i = 0; i < this->size(); ++i) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }

    template <typename T>
    Tensor<T> matmul_2d(const Tensor<T> &a, const Tensor<T> &b) {
        assert(a.get_shape().size() == 2 && b.get_shape().size() == 2 && "Both tensors must be 2D for matmul_2d");
        assert(a.get_shape(1) == b.get_shape(0) && "Inner dimensions must match for matmul_2d");

        size_t m = a.get_shape(0);
        size_t n = a.get_shape(1);
        size_t p = b.get_shape(1);

        Tensor<T> result = Tensor<T>::from_shape({m, p});

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                T sum = 0;
                for (size_t k = 0; k < n; ++k)
                    sum += a(i, k) * b(k, j);

                result(i, j) = sum;
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b) {
        throw std::runtime_error("Generalized matmul not implemented yet.");
    }
}