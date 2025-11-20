#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include <sstream>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <functional>

template<typename T>
struct is_initializer_list : std::false_type {};

template<typename T>
struct is_initializer_list<std::initializer_list<T>> : std::true_type {};

namespace blass {
    template <typename T>
    class Tensor;

    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    Tensor<T> divide(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

    template <typename T>
    Tensor<T> matmul_2d(const Tensor<T>& a, const Tensor<T>& b, bool use_omp = true);

    std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b);
    
    template <char op, typename T>
    void elementwise_op(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& result, 
                        const std::vector<size_t>& shape);

    template <typename T>
    class Tensor {
    private:
        std::shared_ptr<T[]> data;
        std::vector<size_t> shape;
        std::vector<size_t> strides; // strides[i] = product of shape[i+1..end] = amount to walk to next index in dimension i
        size_t sz;

        template <typename U>
        void deduce_shape_from_list(const std::initializer_list<U>& list, std::vector<size_t>& shape_vec) {
            shape_vec.push_back(list.size());
            if (list.size() > 0) {
                if constexpr (is_initializer_list<U>::value) {
                    deduce_shape_from_list(*list.begin(), shape_vec);
                }
            }
        }

        template <typename U>
        void flatten_list_to_data(const std::initializer_list<U>& list, size_t& index) {
            if constexpr (is_initializer_list<U>::value) {
                for (const auto& inner_list : list) {
                    flatten_list_to_data(inner_list, index);
                }
            } 
            else {
                for (const auto& val : list) {
                    data[index++] = static_cast<T>(val);
                }
            }
        }
        
        template <typename ListT>
        void init_from_list(const ListT& list) {
            deduce_shape_from_list(list, shape);

            strides.resize(shape.size());
            if (!shape.empty()) {
                strides.back() = 1;
                for (size_t i = shape.size() - 1; i > 0; --i) {
                    strides[i - 1] = strides[i] * shape[i];
                }
            }

            sz = strides.empty() ? 1 : strides[0] * shape[0];
            data = std::shared_ptr<T[]>(new T[sz]);

            size_t index = 0;
            flatten_list_to_data(list, index);
        }

    public:
        static Tensor<T> from_shape(const std::vector<size_t>& shape_) {
            return Tensor<T>(shape_);
        }

        static Tensor<T> fill(const std::vector<size_t>& shape_, const T& value) {
            Tensor<T> tensor(shape_);
            std::fill_n(tensor.data.get(), tensor.size(), value);
            return tensor;
        }

        static Tensor<T> fill_random(const std::vector<size_t>& shape_, const T& min_value, const T& max_value) {
            Tensor<T> tensor(shape_);
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.data[i] = min_value + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max_value - min_value)));
            }
            return tensor;
        }

        explicit Tensor(const std::vector<size_t>& shape_) : shape(shape_) {
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

        Tensor(const std::shared_ptr<T[]>& data_, const std::vector<size_t>& shape_) : data(data_), shape(shape_) {
            strides.resize(shape.size());
            if (!shape.empty()) {
                strides.back() = 1;
                for (size_t i = shape.size() - 1; i > 0; --i) {
                    strides[i - 1] = strides[i] * shape[i];
                }
            }
            sz = strides.empty() ? 1 : strides[0] * shape[0];
        }

        Tensor(const std::shared_ptr<T[]>& data_, const std::vector<size_t>& shape_, const std::vector<size_t>& strides_)
        : data(data_), shape(shape_), strides(strides_) {
            sz = strides.empty() ? 1 : strides[0] * shape[0];
        }

        template <typename U>
        Tensor(std::initializer_list<U> list) {
            init_from_list(list);
        }
        
        template <typename U>
        Tensor(std::initializer_list<std::initializer_list<U>> list) {
            init_from_list(list);
        }

        template <typename U>
        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<U>>> list) {
            init_from_list(list);
        }
        
        template <typename U>
        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> list) {
            init_from_list(list);
        }

        template <typename U>
        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> list) {
            init_from_list(list);
        }

        template <typename U>
        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>> list) {
            init_from_list(list);
        }

        template <typename U>
        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>> list) {
            init_from_list(list);
        }

        Tensor<T> at(size_t index) const {
            if (shape.empty()) {
                throw std::out_of_range("Cannot index into a scalar tensor");
            }
            if (index >= shape[0]) {
                throw std::out_of_range("Index out of range at dimension 0");
            }
            std::shared_ptr<T[]> slice(data, data.get() + index * strides[0]);
            std::vector<size_t> new_shape(shape.begin() + 1, shape.end());
            std::vector<size_t> new_strides(strides.begin() + 1, strides.end());

            return Tensor<T>(slice, new_shape, new_strides);
        }

        Tensor<T> at(std::initializer_list<size_t> indices) const {
            if (indices.size() > shape.size()) {
                throw std::out_of_range("Too many indices provided");
            }
            size_t offset = 0;
            auto idx_ptr = indices.begin();
            for (size_t i = 0; i < indices.size(); ++i) {
                size_t idx = idx_ptr[i];
                if (idx >= shape[i]) {
                    throw std::out_of_range("Index out of range at dimension " + std::to_string(i));
                }
                offset += idx * strides[i];
            }
            std::shared_ptr<T[]> slice(data, data.get() + offset);
            std::vector<size_t> new_shape(shape.begin() + indices.size(), shape.end());
            std::vector<size_t> new_strides(strides.begin() + indices.size(), strides.end());

            return Tensor<T>(slice, new_shape, new_strides);
        }

        Tensor<T> operator[](size_t index) {
            return at(index);
        }

        Tensor<T> operator[](const std::vector<size_t>& indices) {
            return at(indices);
        }

        template<typename... Indices>
        T& operator()(Indices... indices) {
            if (sizeof...(indices) != shape.size()) {
                throw std::invalid_argument("Incorrect number of indices provided");
            }

            size_t idxs[] = { static_cast<size_t>(indices)... };

            size_t offset = 0;
            for (size_t i = 0; i < shape.size(); i++) {
                if (idxs[i] >= shape[i]) {
                    throw std::out_of_range("Index out of range at dimension " + std::to_string(i));
                }
                offset += idxs[i] * strides[i];
            }

            return data[offset];
        }

        template<typename... Indices>
        const T& operator()(Indices... indices) const {
            if (sizeof...(indices) != shape.size()) {
                throw std::invalid_argument("Incorrect number of indices provided");
            }

            size_t idxs[] = { static_cast<size_t>(indices)... };

            size_t offset = 0;
            for (size_t i = 0; i < shape.size(); i++) {
                if (idxs[i] >= shape[i]) {
                    throw std::out_of_range("Index out of range at dimension " + std::to_string(i));
                }
                offset += idxs[i] * strides[i];
            }

            return data[offset];
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
        T& scalar() const {
            return *data.get();
        }

        bool is_scalar() const {
            return shape.empty();
        }

        const std::vector<size_t>& get_shape() const {
            return shape;
        }

        size_t get_shape(size_t dim) const {
            if (dim < shape.size()) {
                return shape[dim];
            }
            throw std::out_of_range("Dimension out of range");
        }

        const std::vector<size_t>& get_strides() const {
            return strides;
        }

        size_t get_stride(size_t dim) const {
            if (dim < strides.size()) {
                return strides[dim];
            }
            throw std::out_of_range("Dimension out of range");
        }

        const std::shared_ptr<T[]>& get_data_ptr() const {
            return data;
        }

        T* get_data() const {
            return data.get();
        }

        size_t size() const {
            return sz;
        }

        std::string to_string() {
            std::ostringstream oss;
            if (is_scalar()) {
                oss << scalar();
            } else {
                oss << "[";
                for (size_t i = 0; i < shape[0]; i++) {
                    oss << at(i).to_string();
                    if (i < shape[0] - 1) {
                        oss << ", ";
                    }
                }
                oss << "]";
            }
            return oss.str();
        }

        // utilities
        bool is_contiguous() const;
        Tensor<T> contiguous() const;
        Tensor<T> transpose(const std::vector<size_t>& perm) const;
        Tensor<T> transpose() const;
        Tensor<T> view(const std::vector<int>& new_shape) const;
        Tensor<T> clone() const;

        // broadcasting and elementwise operations
        Tensor<T> broadcast(const std::vector<size_t>& target_shape) const;
        template <char op, typename U>
        friend Tensor<U> elementwise_op(const Tensor<U>& a, const Tensor<U>& b);
    
        // arithmetics
        friend Tensor<T> add<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> subtract<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> multiply<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> divide<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> matmul<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> matmul_2d<>(const Tensor<T>& a, const Tensor<T>& b, bool use_omp);

        Tensor<T> operator+(const Tensor<T>& other) const;
        Tensor<T> operator+(const T& scalar) const;

        Tensor<T> operator-(const Tensor<T>& other) const;
        Tensor<T> operator-(const T& scalar) const;

        Tensor<T> operator*(const Tensor<T>& other) const;
        Tensor<T> operator*(const T& scalar) const;

        Tensor<T> operator/(const Tensor<T>& other) const;
        Tensor<T> operator/(const T& scalar) const;
    };
}

#include "tensor_op.h"
#include "tensor_impl.h"