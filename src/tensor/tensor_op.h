#pragma once

#include <cassert>
#include <functional>
#include <iostream>

#include "../utils/utils.h"

namespace blass {
    std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
        size_t len_a = shape_a.size();
        size_t len_b = shape_b.size();
        size_t len_result = std::max(len_a, len_b);
        std::vector<size_t> result_shape(len_result, 1);

        for (size_t i = 0; i < len_result; i++) {
            size_t dim_a = (i < len_result - len_a) ? 1 : shape_a[i - (len_result - len_a)];
            size_t dim_b = (i < len_result - len_b) ? 1 : shape_b[i - (len_result - len_b)];

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::invalid_argument("Shapes cannot be broadcasted: " + utils::to_string_vec(shape_a) + " and " + utils::to_string_vec(shape_b));
            }
            result_shape[i] = std::max(dim_a, dim_b);
        }
        return result_shape;
    }

    template <typename T>
    Tensor<T> Tensor<T>::broadcast(const std::vector<size_t>& target_shape) const {
        std::vector<size_t> broadcasted_stride(target_shape.size(), 0);
        if (target_shape.size() < shape.size()) {
            throw std::invalid_argument("Cannot broadcast tensor of shape " + utils::to_string_vec(shape) + " to target shape " + utils::to_string_vec(target_shape));
        }
        size_t offset = target_shape.size() - shape.size();
        for (size_t i = 0; i < target_shape.size(); i++) {
            size_t current_dim = (i < offset) ? 1 : shape[i - offset];

            if (current_dim == target_shape[i]) 
                broadcasted_stride[i] = (i < offset) ? 0 : strides[i - offset];
            else if (current_dim == 1) 
                broadcasted_stride[i] = 0;
            else 
                throw std::invalid_argument("Cannot broadcast tensor of shape " + utils::to_string_vec(shape) + " to target shape " + utils::to_string_vec(target_shape));
        }
        return Tensor<T>(data, target_shape, broadcasted_stride);
    }
    
    template<char op, typename T>
    Tensor<T> blass::elementwise_op(const Tensor<T>& a_raw, const Tensor<T>& b_raw) {
        std::vector<size_t> shape = broadcast_shape(a_raw.get_shape(), b_raw.get_shape());
        Tensor<T> a = a_raw.broadcast(shape);
        Tensor<T> b = b_raw.broadcast(shape);
        Tensor<T> result = Tensor<T>::from_shape(shape);

        if (shape.size() == 1) {
            #pragma omp parallel for simd
            for (size_t i = 0; i < shape[0]; i++) {
                result.data[i] = utils::scalar_op<op>(a.data[i], b.data[i]);
            }
        }
        else {
            size_t dim_cut = shape.size() - 1;
            for (size_t i = 0; i < shape.size(); i++) {
                if (result.strides[i] == 1) {
                    dim_cut = i;
                    break;
                }
            }
            std::vector<size_t> batch_shape(shape.begin(), shape.begin() + dim_cut);

            size_t batch_size = 1;
            for (size_t dim : batch_shape) 
                batch_size *= dim;

            size_t inner_size = 1;
            for (size_t dim = dim_cut; dim < shape.size(); dim++) 
                inner_size *= shape[dim];
            
            bool omp_batch = batch_size > 16;
            bool omp_inner = inner_size > 1024 && !omp_batch;

            #pragma omp parallel for schedule(static) if(omp_batch)
            for (size_t idx = 0; idx < batch_size; idx++) {
                size_t offset_a = 0, offset_b = 0, offset_res = idx * inner_size;
                size_t tmp = idx;
                
                for (int d = dim_cut - 1; d >= 0; d--) {
                    size_t dim_idx = tmp % batch_shape[d];
                    tmp /= batch_shape[d];
                    offset_a += dim_idx * a.strides[d];
                    offset_b += dim_idx * b.strides[d];
                }

                T* __restrict__ ptr_a = a.data.get() + offset_a;
                T* __restrict__ ptr_b = b.data.get() + offset_b;
                T* __restrict__ ptr_res = result.data.get() + offset_res;

                size_t a_stride = a.strides[dim_cut];
                size_t b_stride = b.strides[dim_cut];

                if (a_stride == 1 && b_stride == 1) {
                    #pragma omp simd if (omp_inner)
                    for (size_t i = 0; i < inner_size; i++) {
                        ptr_res[i] = utils::scalar_op<op>(ptr_a[i], ptr_b[i]);
                    }
                } else if (a_stride == 1 && b_stride == 0) {
                    T b_val = ptr_b[0];
                    
                    #pragma omp simd if (omp_inner)
                    for (size_t i = 0; i < inner_size; i++) {
                        ptr_res[i] = utils::scalar_op<op>(ptr_a[i], b_val);
                    }
                } else if (a_stride == 0 && b_stride == 1) {
                    T a_val = ptr_a[0];

                    #pragma omp simd if (omp_inner)
                    for (size_t i = 0; i < inner_size; i++) {
                        ptr_res[i] = utils::scalar_op<op>(a_val, ptr_b[i]);
                    }
                } else if (a_stride == 0 && b_stride == 0) {
                    T val = utils::scalar_op<op>(ptr_a[0], ptr_b[0]);

                    #pragma omp simd if (omp_inner)
                    for (size_t i = 0; i < inner_size; i++) {
                        ptr_res[i] = val;
                    }
                } else {
                    #pragma omp simd if (omp_inner)
                    for (size_t i = 0; i < inner_size; i++) {
                        ptr_res[i] = utils::scalar_op<op>(ptr_a[i * a_stride], ptr_b[i * b_stride]);
                    }
                }
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() + b.contiguous();
        }

        if (a.get_shape() == b.get_shape()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        }
        return elementwise_op<'+', T> (a, b);
    }

    template <typename T>
    Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() - b.contiguous();
        }

        if (a.get_shape() == b.get_shape()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        }
        return elementwise_op<'-', T> (a, b);
    }

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() * b.contiguous();
        }

        if (a.get_shape() == b.get_shape()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] * b.data[i];
            }
            return result;
        }
        return elementwise_op<'*', T>(a, b);
    }

    template <typename T>
    Tensor<T> divide(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() / b.contiguous();
        }

        if (a.get_shape() == b.get_shape()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] / b.data[i];
            }
            return result;
        }
        return elementwise_op<'/', T>(a, b);
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
    inline Tensor<T> Tensor<T>::operator/(const Tensor<T>& b) const {
        return divide(*this, b);
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
    inline Tensor<T> Tensor<T>::operator/(const T& scalar) const {
        Tensor<T> result = Tensor<T>::from_shape(this->get_shape());
        for (size_t i = 0; i < this->size(); ++i) {
            result.data[i] = this->data[i] / scalar;
        }
        return result;
    }

    template <typename T>
    Tensor<T> matmul_2d(const Tensor<T> &a, const Tensor<T> &b, bool use_omp) {
        if (!a.is_contiguous()) {
            return matmul_2d(a.contiguous(), b, use_omp);
        }

        assert(a.get_shape().size() == 2 && b.get_shape().size() == 2 && "Both tensors must be 2D for matmul_2d");
        assert(a.get_shape(1) == b.get_shape(0) && "Inner dimensions must match for matmul_2d");

        size_t m = a.get_shape(0);
        size_t n = a.get_shape(1);
        size_t p = b.get_shape(1);

        Tensor<T> result = Tensor<T>::from_shape({m, p});

        Tensor<T> b_transposed = b.transpose(); // assumes transpose does automatic contiguous

        if (m * p < 32) {
            for (size_t i = 0; i < m; ++i) {
                T* __restrict__ ptr_res_row = result.data.get() + i * result.strides[0];
                T* __restrict__ ptr_a_row = a.data.get() + i * a.strides[0];

                for (size_t j = 0; j < p; ++j) {
                    T* __restrict__ ptr_b_row = b_transposed.data.get() + j * b_transposed.strides[0];

                    T sum = 0;

                    #pragma omp parallel for simd reduction(+:sum) if (use_omp)
                    for (size_t k = 0; k < n; ++k)
                        sum += ptr_a_row[k] * ptr_b_row[k];

                    ptr_res_row[j] = sum;
                }
            }
        }
        else if (m >= p) {
            #pragma omp parallel for if (use_omp)
            for (size_t i = 0; i < m; ++i) {
                T* __restrict__ ptr_res_row = result.data.get() + i * result.strides[0];
                T* __restrict__ ptr_a_row = a.data.get() + i * a.strides[0];

                for (size_t j = 0; j < p; ++j) {
                    T* __restrict__ ptr_b_row = b_transposed.data.get() + j * b_transposed.strides[0];

                    T sum = 0;

                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < n; ++k)
                        sum += ptr_a_row[k] * ptr_b_row[k];

                    ptr_res_row[j] = sum;
                }
            }
        } 
        else {
            #pragma omp parallel for if (use_omp)
            for (size_t j = 0; j < p; ++j) {
                T* __restrict__ ptr_res_col = result.data.get() + j;
                T* __restrict__ ptr_b_row = b_transposed.data.get() + j * b_transposed.strides[0];
                
                for (size_t i = 0; i < m; ++i) {
                    T* __restrict__ ptr_a_row = a.data.get() + i * a.strides[0];

                    T sum = 0;

                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < n; ++k)
                        sum += ptr_a_row[k] * ptr_b_row[k];

                    ptr_res_col[i * result.strides[0]] = sum;
                }
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b) {
        if (a.shape.size() < 2 || b.shape.size() < 2) {
            throw std::invalid_argument("Both tensors must be at least 2D for matmul");
        }

        if (a.shape.size() == 2 && b.shape.size() == 2) {
            return matmul_2d(a, b);
        }

        std::vector<size_t> batch_a_shape, batch_b_shape;
        for (size_t i = 0; i < a.shape.size() - 2; i++)
            batch_a_shape.push_back(a.shape[i]);

        for (size_t i = 0; i < b.shape.size() - 2; i++)
            batch_b_shape.push_back(b.shape[i]);

        if (batch_a_shape.size() == 0)
            batch_a_shape.push_back(1);

        if (batch_b_shape.size() == 0)
            batch_b_shape.push_back(1);
        
        std::vector<size_t> batch_result_shape = broadcast_shape(batch_a_shape, batch_b_shape);
        if (a.shape[a.shape.size() - 1] != b.shape[b.shape.size() - 2]) {
            throw std::invalid_argument("Inner dimensions must match for matmul");
        }

        std::vector<size_t> a_broadcast_shape = batch_result_shape;
        a_broadcast_shape.push_back(a.shape[a.shape.size() - 2]);
        a_broadcast_shape.push_back(a.shape[a.shape.size() - 1]);

        std::vector<size_t> b_broadcast_shape = batch_result_shape;
        b_broadcast_shape.push_back(b.shape[b.shape.size() - 2]);
        b_broadcast_shape.push_back(b.shape[b.shape.size() - 1]);

        Tensor<T> a_broadcasted = a.broadcast(a_broadcast_shape);
        Tensor<T> b_broadcasted = b.broadcast(b_broadcast_shape);

        std::vector<size_t> result_shape = batch_result_shape;
        result_shape.push_back(a.shape[a.shape.size() - 2]);
        result_shape.push_back(b.shape[b.shape.size() - 1]);

        Tensor<T> result = Tensor<T>::from_shape(result_shape);

        size_t batch_size = 1;

        for (size_t dim : batch_result_shape)
            batch_size *= dim;

        size_t batch_stride = a.shape[a.shape.size() - 2] * b.shape[b.shape.size() - 1];
        
        const auto& sa = a_broadcasted.strides;
        const auto& sb = b_broadcasted.strides;

        bool use_omp_batch = batch_size >= 32;

        #pragma omp parallel for if(use_omp_batch)
        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<size_t> idx(batch_result_shape.size(), 0);

            size_t tmp = i;
            size_t offset_a = 0, offset_b = 0;
            for (int dim = batch_result_shape.size() - 1; dim >= 0; --dim) {
                size_t idx = tmp % batch_result_shape[dim];
                offset_a += idx * sa[dim];
                offset_b += idx * sb[dim];
                tmp /= batch_result_shape[dim];
            }

            std::shared_ptr<T[]> ptr_a = std::shared_ptr<T[]>(a_broadcasted.data, a_broadcasted.data.get() + offset_a);
            std::shared_ptr<T[]> ptr_b = std::shared_ptr<T[]>(b_broadcasted.data, b_broadcasted.data.get() + offset_b);
            T* __restrict__ ptr_res = result.data.get() + i * batch_stride;

            Tensor<T> a_slice(ptr_a, 
                            {a.shape[a.shape.size() - 2], a.shape[a.shape.size() - 1]});
            Tensor<T> b_slice(ptr_b, 
                                {b.shape[b.shape.size() - 2], b.shape[b.shape.size() - 1]});

            Tensor<T> mat_result = matmul_2d(a_slice, b_slice, !use_omp_batch);
            T* __restrict__ mat_result_ptr = mat_result.data.get();
            
            #pragma omp simd
            for (size_t j = 0; j < mat_result.size(); ++j)
                ptr_res[j] = mat_result_ptr[j];
        }
        return result;
    }
}