#pragma once 

#include <stdfloat>
#include "../tensor/tensor.h"
#include "modules.h"
#include "gguf_reader.h"
#include "tokenizer.h"

namespace blass {
    namespace models {
        class Qwen2Block: public nn::Module<float> {
            // attn_norm_weight: float32, attn_k_weight: float16, attn_q_weight: float16
            // attn_v_weight: float16, attn_output_weight: float16
            
            // attn_k_bias: float32, attn_q_bias: float32, attn_v_bias: float32
            // ffn_norm: float32, ffn_down: float16, ffn_gate: float16, ffn_up: float16
            std::map<std::string, Tensor<float>> tensors;

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

            void load_tensor_f16(Tensor<float> &param, const gguf_loader::tensor_data &data) {
                std::vector<size_t> dims(data.dims.begin(), data.dims.end());
                param = Tensor<float>::from_shape(dims);
                size_t total_elems = param.size();
                float* dest = param.get_data();
                uint16_t* src = (uint16_t*)data.data;
                for (size_t i = 0; i < total_elems; i++) {
                    dest[i] = std::float16_t(src[i]);
                }
            }

            void load_tensor_f32(Tensor<float> &param, const gguf_loader::tensor_data &data) {
                std::vector<size_t> dims(data.dims.begin(), data.dims.end());
                param = Tensor<float>::from_shape(dims);
                size_t total_elems = param.size();
                float* dest = param.get_data();
                float* src = (float*)data.data;
                std::memcpy(dest, src, total_elems * sizeof(float));
            }

            void init_param(std::string name, const gguf_loader::tensor_data &data) {
                Tensor<float> param;
                if (data.type == gguf_loader::GGML_TYPE_F16) {
                    load_tensor_f16(param, data);
                } 
                else if (data.type == gguf_loader::GGML_TYPE_F32) {
                    load_tensor_f32(param, data);
                } 
                else 
                    throw std::runtime_error("Unsupported tensor type in Qwen2Block: " + std::to_string((uint32_t)data.type));
                
                if (name == "attn_norm.weight") {
                    attn_norm->load_weight(param);
                }
                else if (name == "ffn_norm.weight")
                    ffn_norm->load_weight(param);
                else {
                    tensors[name] = param.clone();
                    register_parameter(name, tensors[name]);
                }
            }

            Tensor<float> forward(const Tensor<float>& input) override {
                int num_attn_heads = 14;
                int num_kv_heads = 2;
                int head_dim = 64;
                int hidden_size = 896;
                int batch_size = input.get_shape(0);
                int seq_len = input.get_shape(1);
                
                Tensor<float> residual = input.clone();

                Tensor<float> x = (*attn_norm)(input);

                Tensor<float> q = matmul(x, *params["attn_q.weight"], true) + *params["attn_q.bias"];
                Tensor<float> k = matmul(x, *params["attn_k.weight"], true) + *params["attn_k.bias"];
                Tensor<float> v = matmul(x, *params["attn_v.weight"], true) + *params["attn_v.bias"];
                
                // TODO: RoPE

                q = q.view({batch_size, seq_len, num_attn_heads, head_dim}).contiguous();
                k = k.view({batch_size, seq_len, num_kv_heads, 1, head_dim}).contiguous();
                v = v.view({batch_size, seq_len, num_kv_heads, 1, head_dim}).contiguous();

                k = k.broadcast({(size_t)batch_size, (size_t)seq_len, (size_t)num_kv_heads, (size_t)(num_attn_heads / num_kv_heads), (size_t)head_dim}).contiguous();
                v = v.broadcast({(size_t)batch_size, (size_t)seq_len, (size_t)num_kv_heads, (size_t)(num_attn_heads / num_kv_heads), (size_t)head_dim}).contiguous();

                k = k.view({batch_size, seq_len, num_attn_heads, head_dim});
                v = v.view({batch_size, seq_len, num_attn_heads, head_dim});

                q = q.transpose({0, 2, 1, 3}); 
                k = k.transpose({0, 2, 1, 3}); 
                v = v.transpose({0, 2, 1, 3});

                Tensor<float> scores = matmul(q, k.transpose2D()) / sqrt(64.0f);
                Tensor<float> attn_weights = nn::functional::softmax(scores);
                Tensor<float> attn_out = matmul(attn_weights, v);

                attn_out = attn_out.transpose({0, 2, 1, 3});
                attn_out = attn_out.contiguous().view({batch_size, seq_len, hidden_size});

                x = matmul(attn_out, *params["attn_output.weight"], true);
                x = x + residual;

                residual = x;

                x = (*ffn_norm)(x);
                Tensor<float> gate = matmul(x, *params["ffn_gate.weight"], true);
                Tensor<float> up = matmul(x, *params["ffn_up.weight"], true);

                Tensor<float> activated = (*ffn_activation)(gate) * up;

                x = matmul(activated, *params["ffn_down.weight"], true);
                x = x + residual;

                return x;
            }
        };

        class Qwen2Model: public nn::Module<float> {
            std::shared_ptr<Qwen2Block> blocks[24];
            std::shared_ptr<nn::RMSNorm<float>> output_norm;

            Tensor<float> token_embd;

        public:
            tokenizer::Tokenizer tk;

            Qwen2Model() {
                for (int i = 0; i < 24; i++) {
                    blocks[i] = std::make_shared<Qwen2Block>();
                    this->register_module("block_" + std::to_string(i), blocks[i]);
                }
                output_norm = std::make_shared<nn::RMSNorm<float>>(1, 1e-6f);
                this->register_module("output_norm", output_norm);
            }

            void load_tensor_f16(Tensor<float> &param, const gguf_loader::tensor_data &data) {
                std::vector<size_t> dims(data.dims.begin(), data.dims.end());
                param = Tensor<float>::from_shape(dims);
                size_t total_elems = param.size();
                float* dest = param.get_data();
                uint16_t* src = (uint16_t*)data.data;
                for (size_t i = 0; i < total_elems; i++) {
                    dest[i] = std::float16_t(src[i]);
                }
            }

            void load_tensor_f32(Tensor<float> &param, const gguf_loader::tensor_data &data) {
                std::vector<size_t> dims(data.dims.begin(), data.dims.end());
                param = Tensor<float>::from_shape(dims);
                size_t total_elems = param.size();
                float* dest = param.get_data();
                float* src = (float*)data.data;
                std::memcpy(dest, src, total_elems * sizeof(float));
            }

            void load_model(const char* filepath) {
                gguf_loader::GGUFModel model(filepath);
                tk = model.tk;
                for (auto &tensor_pair : model.tensors) {
                    std::string name = tensor_pair.first;
                    auto &tensor_info = tensor_pair.second;

                    std::cout << "Loading tensor: " << name << std::endl;

                    if (name.substr(0, 3) == "blk") {
                        int block_idx = 0;
                        int suf_idx = 0;
                        bool dot = 0;
                        for (size_t i = 4; i < name.size(); i++) {
                            if (name[i] == '.') {
                                dot = 1;
                                continue;
                            }
                            if (!dot)
                                block_idx = block_idx * 10 + (name[i] - '0');
                            else {
                                suf_idx = i;
                                break;
                            }
                        }

                        std::string param_name = name.substr(suf_idx);
                        blocks[block_idx]->init_param(param_name, tensor_info);
                    }
                    else {
                        // todo: load output_norm and other params
                        Tensor<float> param;
                        if (tensor_info.type == gguf_loader::GGML_TYPE_F16) {
                            load_tensor_f16(param, tensor_info);
                        } 
                        else if (tensor_info.type == gguf_loader::GGML_TYPE_F32) {
                            load_tensor_f32(param, tensor_info);
                        } 
                        else 
                            throw std::runtime_error("Unsupported tensor type in Qwen2Model: " + std::to_string((uint32_t)tensor_info.type));
                        if (name == "output_norm.weight")
                            output_norm->load_weight(param);
                        else if (name == "token_embd.weight") {
                            token_embd = param.clone();
                            register_parameter(name, token_embd);
                        }
                    }
                }
            }

            Tensor<float> run_inference(const std::vector<int>& token_ids) {
                size_t seq_len = token_ids.size();
                int hidden_dim = params["token_embd.weight"]->get_shape(1);

                Tensor<float> x = Tensor<float>::from_shape({(size_t)1, seq_len, (size_t)hidden_dim});
                std::cout << "Running inference with input shape: " << utils::to_string_vec(x.get_shape()) << std::endl;

                float* embedding_data = params["token_embd.weight"]->get_data();
                float* input_data = x.get_data();

                for (size_t i = 0; i < seq_len; i++) {
                    int token = token_ids[i];

                    float* src_row = embedding_data + token * hidden_dim;
                    float* dest_row = input_data + i * hidden_dim;
                    
                    std::memcpy(dest_row, src_row, hidden_dim * sizeof(float));
                }

                return forward(x);
            }

            Tensor<float> forward(const Tensor<float>& input) override {
                Tensor<float> x = input.clone();

                std::cout << "Starting forward pass through Qwen2Model..." << std::endl;

                for (int i = 0; i < 24; i++) {
                    x = (*blocks[i])(x);
                }

                x = (*output_norm)(x);
                return x;
            }
        };
    };
};