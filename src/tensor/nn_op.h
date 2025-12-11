#pragma once

#include <math.h>

namespace blass {
    namespace nn {
        template<typename T>
        Tensor<T> softmax(const Tensor<T>& input) {
            Tensor<T> result = input.clone(); // result should be contiguous 
            size_t last_dim = result.get_shape().back();
            size_t batch_size = result.size() / last_dim;

            T* data = result.get_data();
            
            #pragma omp parallel for
            for (size_t i = 0; i < batch_size; i++) {
                T* row = data + i * last_dim;

                T max_val = std::numeric_limits<T>::lowest();
                for (size_t j = 0; j < last_dim; j++)
                    if (max_val < row[j]) max_val = row[j];

                T sum = 0;
                for (size_t j = 0; j < last_dim; j++) {
                    row[j] = std::exp(row[j] - max_val);
                    sum += row[j];
                }

                T inv_sum = static_cast<T>(1.0) / sum;
                for (size_t j = 0; j < last_dim; j++)
                    row[j] *= inv_sum;
            }
            return result;
        }
    };
};