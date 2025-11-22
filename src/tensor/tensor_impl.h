#pragma once

#include <cassert>
#include <functional>
#include <iostream>

#include "../utils/utils.h"

namespace blass {
    template <typename T>
    bool Tensor<T>::is_contiguous() const {
        size_t expected_stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            if (strides[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape[i];
        }
        return true;
    }

    template <typename T>
    Tensor<T> Tensor<T>::contiguous() const {
        if (is_contiguous()) {
            return *this;
        }
        Tensor<T> result(shape);

        const size_t ndim = shape.size();
        const size_t total = sz;

        std::vector<size_t> dim_prod(ndim, 1);

        if (ndim > 0) {
            for (int i = ndim - 2; i >= 0; --i)
                dim_prod[i] = dim_prod[i + 1] * shape[i + 1];
        }

        #pragma omp parallel 
        {
            std::vector<size_t> local_multi(ndim);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < total; ++i) {
                size_t tmp = i;
                size_t src_offset = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    size_t idx = (dim_prod[d] == 0) ? 0 : (tmp / dim_prod[d]);
                    tmp -= idx * (dim_prod[d] == 0 ? 0 : dim_prod[d]);
                    src_offset += idx * strides[d];
                }
                result.data[i] = data[src_offset];
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(const std::vector<size_t>& perm) const {
        std::vector<bool> mark(shape.size(), 0);
        for (size_t i = 0; i < perm.size(); i++) 
            mark[perm[i]] = 1;

        if (std::accumulate(mark.begin(), mark.end(), 0) != (int)shape.size())
            throw std::invalid_argument("Invalid permutation vector");
        
        Tensor<T> result(data, shape, strides);

        for (size_t i = 0; i < perm.size(); i++) {
            result.shape[i] = shape[perm[i]];
            result.strides[i] = strides[perm[i]];
        }
        return result.contiguous();
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose() const {
        std::vector<size_t> perm(shape.size());
        for (size_t i = 0; i < shape.size(); i++) {
            perm[i] = shape.size() - 1 - i;
        }
        return transpose(perm);
    }

    template <typename T>
    Tensor<T> Tensor<T>::view(const std::vector<int>& new_shape) const {
        std::vector<int> final_shape = new_shape;
        size_t new_sz = 1;

        // Handle negative dimension -> Infer the size
        int idx_neg = -1;
        for (size_t i = 0; i < final_shape.size(); i++) {
            if (final_shape[i] < -0) {
                if (idx_neg != -1) {
                    throw std::invalid_argument("Only one dimension can be negative in view for dimension inference");
                }
                idx_neg = i;
                continue;
            }
            new_sz *= final_shape[i];
        }
        if (idx_neg != -1) {
            if (sz % new_sz != 0) {
                throw std::invalid_argument("Cannot infer dimension size in view");
            }
            size_t inferred_size = sz / new_sz;
            final_shape[idx_neg] = inferred_size;
        } else if (new_sz != sz) {
            throw std::invalid_argument("Total size must remain unchanged in view");
        }
        return Tensor<T>(contiguous().data, std::vector<size_t>(final_shape.begin(), final_shape.end()));
    }

    template <typename T>
    Tensor<T> Tensor<T>::clone() const {
        Tensor<T> result(shape);
        if (is_contiguous()) {
            std::copy(data.get(), data.get() + sz, result.data.get());
            return result;
        } else {
            for (size_t idx = 0; idx < sz; idx++) {
                size_t offset = 0;
                size_t tmp = idx;

                for (int d = shape.size() - 1; d >= 0; --d) {
                    size_t dim_idx = tmp % shape[d];
                    tmp /= shape[d];
                    offset += dim_idx * strides[d];
                }
                result.data[idx] = data[offset];
            }

            return result;
        }
    }
}