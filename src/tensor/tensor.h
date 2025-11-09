#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>

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
        T& scalar() const {
            return *data.get();
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
    
        // arithmetics
        friend Tensor<T> add<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> subtract<>(const Tensor<T>& a, const Tensor<T>& b);
        friend Tensor<T> multiply<>(const Tensor<T>& a, const Tensor<T>& b);

        Tensor<T> operator+(const Tensor<T>& other) const;
        Tensor<T> operator+(const T& scalar) const;

        Tensor<T> operator-(const Tensor<T>& other) const;
        Tensor<T> operator-(const T& scalar) const;

        Tensor<T> operator*(const Tensor<T>& other) const;
        Tensor<T> operator*(const T& scalar) const;
    };
}

#include "tensor_impl.h"