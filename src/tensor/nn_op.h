#pragma once

#include <math.h>
#include <map>

namespace blass {
    namespace nn {
        namespace functional {
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
        }

        template <typename T>
        class Module {
        protected:
            std::map<std::string, Tensor<T>*> params;
            std::map<std::string, std::shared_ptr<Module<T>>> modules;
            std::string name;

        public:
            virtual ~Module() = default;

            Tensor<T> operator()(const Tensor<T>& input) {
                return forward(input);
            }

            virtual Tensor<T> forward(const Tensor<T>& input) = 0;

            void register_parameter(std::string key, Tensor<T> &param) {
                params[key] = &param;
            }

            template <typename U>
            void register_module(std::string key, std::shared_ptr<U> module) {
                modules[key] = module;
            }
        };

        template <typename T>
        class Linear : public Module<T> {
        public:
            Tensor<T> weight;
            Tensor<T> bias;

            Linear(int in, int out) {
                weight = Tensor<T>::fill_random({in, out}, -0.1f, 0.1f);
                bias = Tensor<T>::fill_random({out}, -0.1f, 0.1f);

                this->register_parameter("weight", weight);
                this->register_parameter("bias", bias);
            }

            Tensor<T> forward(const Tensor<T>& input) override {
                Tensor<T> output = matmul(input, weight) ;
                output = output + bias;
                return output;
            }
        };

        template <typename T>
        class Softmax : public Module<T> {
        public:
            Softmax() {}

            Tensor<T> forward(const Tensor<T>& input) override {
                return functional::softmax(input);
            }
        };
    };
};