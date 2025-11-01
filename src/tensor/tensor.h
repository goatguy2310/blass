#pragma once

#include <vector>
#include <memory>
#include <numeric>

template <typename T>
class Tensor {
private:
    std::shared_ptr<T[]> data;
    std::vector<size_t> shape;
    size_t sz;

public:

    Tensor(const std::vector<size_t>& shape_) : shape(shape_) {
        sz = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        data = std::shared_ptr<T[]>(new T[sz]);
    }

    Tensor(const std::shared_ptr<T[]>& data_, const std::vector<size_t>& shape_)
        : data(data_), shape(shape_) {
        sz = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    }

    Tensor operator[](size_t index) {
        size_t width = sz / shape[0];

        std::shared_ptr<T[]> slice(data, data.get() + index * width);
        std::vector<size_t> new_shape(shape.begin() + 1, shape.end());

        return Tensor<T>(slice, new_shape);
    }

    Tensor<T> operator=(T value) {
        for (size_t i = 0; i < sz; ++i) {
            data[i] = value;
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