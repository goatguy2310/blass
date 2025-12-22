#pragma once 

#include <stdfloat>
#include "nn/modules.h"
#include "nn/gguf_reader.h"

namespace blass {
    namespace models {
        class Qwen2Model: public nn::Module<float> {
            std::shared_ptr<Qwen2Block> blocks[23];
            std::shared_ptr<nn::RMSNorm<float>> output_norm;
        public:
            Qwen2Model() {
                for (int i = 0; i < 23; i++) {
                    blocks[i] = std::make_shared<Qwen2Block>();
                    this->register_module("block_" + std::to_string(i), blocks[i]);
                }
                output_norm = std::make_shared<nn::RMSNorm<float>>(1, 1e-6f);
                this->register_module("output_norm", output_norm);
            }
            void load_model(const char* filepath) {
                gguf_loader::GGUFModel model(filepath);
                for (auto &tensor_pair : model.tensors) {
                    std::string name = tensor_pair.first;
                    auto &tensor_info = tensor_pair.second;

                    if (std::string_view(name, 3) == "blk") {
                        int block_idx = 0;
                        int suf_idx = 0;
                        bool dot = 0;
                        for (int i = 0; i < name.size(); i++) {
                            if (name[i] == '.') {
                                dot = 1;
                                continue;
                            }
                            if (!dot)
                                block_idx = block_idx * 10 + (name[i] - '0');
                            else {
                                suf_idx = i + 1;
                                break;
                            }
                        }

                        std::string param_name = name.substr(suf_idx);
                        blocks[block_idx]->init_param(param_name, tensor_info);
                    }
                    else {
                        // todo: load output_norm and other params
                    }
                }
            }

            void forward(const Tensor<float>& input) override {
                Tensor<float> x = input;
                for (int i = 0; i < 23; i++) {
                    x = (*blocks[i])(x);
                }

                x = (*output_norm)(x);
                return x;
            }
        };

        class Qwen2Block: public nn::Module<float> {
            // attn_norm_weight: float32, attn_k_weight: float16, attn_q_weight: float16
            // attn_v_weight: float16, attn_output_weight: float16
            
            // attn_k_bias: float32, attn_q_bias: float32, attn_v_bias: float32
            // ffn_norm: float32, ffn_down: float16, ffn_gate: float16, ffn_up: float16

            std::shared_ptr<nn::RMSNorm<float>> attn_norm, ffn_norm;
            std::shared_ptr<nn::SiLU<float>> ffn_activation;
        public:
            Qwen2Block() {
                attn_norm = std::make_shared<nn::RMSNorm<float>>(4096, 1e-6f);
                ffn_norm = std::make_shared<nn::RMSNorm<float>>(4096, 1e-6f);
                ffn_activation = std::make_shared<nn::SiLU<float>>();
                this->register_module("attn_norm", attn_norm);
                this->register_module("ffn_norm", ffn_norm);
                this->register_module("ffn_activation", ffn_activation);
            }

            void load_tensor_f16(Tensor<float> &param, const tensor_data &data) {
                param = Tensor<float>::from_shape({data.dims});
                size_t total_elems = param.size();
                float* dest = param.get_data();
                uint16_t* src = (uint16_t*)data.data;
                for (size_t i = 0; i < total_elems; i++) {
                    dest[i] = std::float16_t(src[i]);
                }
            }

            void load_tensor_f32(Tensor<float> &param, const tensor_data &data) {
                param = Tensor<float>::from_shape({data.dims});
                size_t total_elems = param.size();
                float* dest = param.get_data();
                float* src = (float*)data.data;
                std::memcpy(dest, src, total_elems * sizeof(float));
            }

            void init_param(std::string name, const tensor_data &data) {
                Tensor<float> param;
                if (data.type == gguf_loader::GGML_TYPE_F16) {
                    load_tensor_f16(param, data);
                } 
                else if (data.type == gguf_loader::GGML_TYPE_F32) {
                    load_tensor_f32(param, data);
                } 
                else 
                    throw std::runtime_error("Unsupported tensor type in Qwen2Block: " + std::to_string((uint32_t)data.type));

                if (name == "attn_norm.weight")
                    attn_norm->load_weight(param);
                else if (name == "ffn_norm.weight")
                    ffn_norm->load_weight(param);
                else
                    register_parameter(name, param);
            }

            Tensor<float> forward(const Tensor<float>& input) override {
                Tensor<float> residual = input;

                Tensor<float> x = (*attn_norm)(input);

                Tensor<float> q = matmul(x, params["attn_q.weight"]) + attn_q_bias;
                Tensor<float> k = matmul(x, params["attn_k.weight"]) + attn_k_bias;
                Tensor<float> v = matmul(x, params["attn_v.weight"]) + attn_v_bias;
                // RoPE

                Tensor<float> scores = matmul(q, k.transpose()) / sqrt(64.0f);
                Tensor<float> attn_weights = functional::softmax(scores);
                Tensor<float> attn_out = matmul(attn_weights, v);

                x = matmul(attn_out, params["attn_output.weight"]);
                x = x + residual;

                residual = x;

                x = (*ffn_norm)(x);
                Tensor<float> gate = matmul(x, params["ffn_gate.weight"]);
                Tensor<float> up = matmul(x, params["ffn_up.weight"]);

                Tensor<float> activated = (*ffn_activation)(gate) * up;

                x = matmul(activated, params["ffn_down.weight"]);
                x = x + residual;

                return x;
            }
        }
    };
};