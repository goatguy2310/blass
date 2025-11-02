#pragma once

#include <vector>
#include <memory>
#include <numeric>

template <typename T>
class Tensor {
private:
    std::shared_ptr<T[]> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides; // strides[i] = product of shape[i+1..end] = amount to walk to next index in dimension i
    size_t sz;

public:

    Tensor(const std::vector<size_t>& shape_) : shape(shape_) {
        strides.resize(shape.size());
        if (!shape.empty()) {
            strides.back() = 1;
            for (size_t i = shape.size() - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
            }
        }
        sz = strides.empty() ? 1 : strides[0] * shape[0];
        data = std::shared_ptr<T[]>(new T[sz]);
    }

    Tensor(const std::shared_ptr<T[]>& data_, const std::vector<size_t>& shape_)
        : data(data_), shape(shape_) {
        strides.resize(shape.size());
        if (!shape.empty()) {
            strides.back() = 1;
            for (size_t i = shape.size() - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
            }
        }
        sz = strides.empty() ? 1 : strides[0] * shape[0];
    }

    Tensor<T> operator[](size_t index) {
        std::shared_ptr<T[]> slice(data, data.get() + index * strides[0]);
        std::vector<size_t> new_shape(shape.begin() + 1, shape.end());

        return Tensor<T>(slice, new_shape);
    }

    Tensor<T>& operator=(T scalar) {
        for (size_t i = 0; i < sz; ++i) {
            data[i] = scalar;
        }
        return *this;
    }

    /**
     * Returns the first element of the tensor.
     * Use this to get the value of a scalar tensor.
     */
    T scalar() const {
        return data[0];
    }

    bool is_scalar() const {
        return shape.empty();
    }

    std::vector<size_t> get_shape() const {
        return shape;
    }

    size_t get_shape(size_t dim) const {
        if (dim < shape.size()) {
            return shape[dim];
        }
        throw std::out_of_range("Dimension out of range");
    }

    size_t size() const {
        return sz;
    }
};