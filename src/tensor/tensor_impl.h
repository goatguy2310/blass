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
            result[i].scalar() = a[i].scalar() - b[i].scalar();
        }
        return result;
    }

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for multiplication");
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i].scalar() = a[i].scalar() * b[i].scalar();
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
}