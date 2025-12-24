#pragma once

#include <iostream>
#include <variant>
#include <cstdint>
#include <vector>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include "tokenizer.h"

namespace blass {
    namespace gguf_loader {
        class MemoryMappedFile {
        public:
            void* data;
            size_t size;

            MemoryMappedFile(const char* filepath) {
                int fd = open(filepath, O_RDONLY);
                if (fd == -1) throw std::runtime_error("Could not open file");

                struct stat sb;
                if (fstat(fd, &sb) == -1) {
                    close(fd);
                    throw std::runtime_error("Could not get file size");
                }
                size = sb.st_size;

                data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
                
                close(fd);

                if (data == MAP_FAILED) {
                    throw std::runtime_error("mmap failed");
                }
            }
            
            template<typename T>
            T read_at(size_t offset) {
                if (offset + sizeof(T) > size) {
                    throw std::runtime_error("Read out of bounds in MemoryMappedFile");
                }
                T out;
                std::memcpy(&out, (char*)data + offset, sizeof(T));
                return out;
            }

            std::string read_string_at(size_t offset, size_t length) {
                if (offset + length > size) {
                    throw std::runtime_error("Read out of bounds in MemoryMappedFile");
                }
                return std::string((char*)data + offset, length);
            }

            ~MemoryMappedFile() {
                if (data != MAP_FAILED) {
                    munmap(data, size);
                }
            }
        };

        enum ggml_type: uint32_t {
            GGML_TYPE_F32     = 0,
            GGML_TYPE_F16     = 1,
            GGML_TYPE_Q4_0    = 2,
            GGML_TYPE_Q4_1    = 3,
            // GGML_TYPE_Q4_2 = 4, support has been removed
            // GGML_TYPE_Q4_3 = 5, support has been removed
            GGML_TYPE_Q5_0    = 6,
            GGML_TYPE_Q5_1    = 7,
            GGML_TYPE_Q8_0    = 8,
            GGML_TYPE_Q8_1    = 9,
            GGML_TYPE_Q2_K    = 10,
            GGML_TYPE_Q3_K    = 11,
            GGML_TYPE_Q4_K    = 12,
            GGML_TYPE_Q5_K    = 13,
            GGML_TYPE_Q6_K    = 14,
            GGML_TYPE_Q8_K    = 15,
            GGML_TYPE_IQ2_XXS = 16,
            GGML_TYPE_IQ2_XS  = 17,
            GGML_TYPE_IQ3_XXS = 18,
            GGML_TYPE_IQ1_S   = 19,
            GGML_TYPE_IQ4_NL  = 20,
            GGML_TYPE_IQ3_S   = 21,
            GGML_TYPE_IQ2_S   = 22,
            GGML_TYPE_IQ4_XS  = 23,
            GGML_TYPE_I8      = 24,
            GGML_TYPE_I16     = 25,
            GGML_TYPE_I32     = 26,
            GGML_TYPE_I64     = 27,
            GGML_TYPE_F64     = 28,
            GGML_TYPE_IQ1_M   = 29,
            GGML_TYPE_BF16    = 30,
            // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
            // GGML_TYPE_Q4_0_4_8 = 32,
            // GGML_TYPE_Q4_0_8_8 = 33,
            GGML_TYPE_TQ1_0   = 34,
            GGML_TYPE_TQ2_0   = 35,
            // GGML_TYPE_IQ4_NL_4_4 = 36,
            // GGML_TYPE_IQ4_NL_4_8 = 37,
            // GGML_TYPE_IQ4_NL_8_8 = 38,
            GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
            GGML_TYPE_COUNT   = 40,
        };

        #define QK_K 256
        struct block_q4_K {
            union {
                struct {
                    uint16_t d;
                    uint16_t dmin;
                };
            };

            uint8_t scales[12];
            uint8_t qk[QK_K / 2];
        };

        enum gguf_metadata_value_type: uint32_t {
            // The value is a 8-bit unsigned integer.
            GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
            // The value is a 8-bit signed integer.
            GGUF_METADATA_VALUE_TYPE_INT8 = 1,
            // The value is a 16-bit unsigned little-endian integer.
            GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
            // The value is a 16-bit signed little-endian integer.
            GGUF_METADATA_VALUE_TYPE_INT16 = 3,
            // The value is a 32-bit unsigned little-endian integer.
            GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
            // The value is a 32-bit signed little-endian integer.
            GGUF_METADATA_VALUE_TYPE_INT32 = 5,
            // The value is a 32-bit IEEE754 floating point number.
            GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
            // The value is a boolean.
            // 1-byte value where 0 is false and 1 is true.
            // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
            GGUF_METADATA_VALUE_TYPE_BOOL = 7,
            // The value is a UTF-8 non-null-terminated string, with length prepended.
            GGUF_METADATA_VALUE_TYPE_STRING = 8,
            // The value is an array of other values, with the length and type prepended.
            ///
            // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
            GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
            // The value is a 64-bit unsigned little-endian integer.
            GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
            // The value is a 64-bit signed little-endian integer.
            GGUF_METADATA_VALUE_TYPE_INT64 = 11,
            // The value is a 64-bit IEEE754 floating point number.
            GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
        };

        struct GGUFMetadataValue {
            gguf_metadata_value_type type;
            std::variant<
                uint8_t,
                int8_t,
                uint16_t,
                int16_t,
                uint32_t,
                int32_t,
                float,
                bool,
                std::string,
                std::vector<GGUFMetadataValue>,
                uint64_t,
                int64_t,
                double
            > value;

            GGUFMetadataValue() {}
            GGUFMetadataValue(MemoryMappedFile &file, uint64_t &offset, int _type = -1) {
                if (_type != -1) {
                    type = (gguf_metadata_value_type)_type;
                } 
                else {
                    type = file.read_at<gguf_metadata_value_type>(offset);
                    offset += 4;
                }

                switch (type) {
                    case GGUF_METADATA_VALUE_TYPE_UINT8: {
                        value = file.read_at<uint8_t>(offset);
                        offset += 1;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT8: {
                        value = file.read_at<int8_t>(offset);
                        offset += 1;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_UINT16: {
                        value = file.read_at<uint16_t>(offset);
                        offset += 2;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT16: {
                        value = file.read_at<int16_t>(offset);
                        offset += 2;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_UINT32: {
                        value = file.read_at<uint32_t>(offset);
                        offset += 4;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT32: {
                        value = file.read_at<int32_t>(offset);
                        offset += 4;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
                        value = file.read_at<float>(offset);
                        offset += 4;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_BOOL: {
                        value = file.read_at<uint8_t>(offset) != 0;
                        offset += 1;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_STRING: {
                        uint64_t length = file.read_at<uint64_t>(offset);
                        offset += 8;
                        value = file.read_string_at(offset, length);
                        offset += length;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_UINT64: {
                        value = file.read_at<uint64_t>(offset);
                        offset += 8;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_INT64: {
                        value = file.read_at<int64_t>(offset);
                        offset += 8;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
                        value = file.read_at<double>(offset);
                        offset += 8;
                        break;
                    }
                    case GGUF_METADATA_VALUE_TYPE_ARRAY: {
                        gguf_metadata_value_type elem_type = file.read_at<gguf_metadata_value_type>(offset);
                        offset += 4;

                        uint64_t length = file.read_at<uint64_t>(offset);
                        offset += 8;

                        std::vector<GGUFMetadataValue> vec;
                        vec.reserve(length);
                        for (uint64_t i = 0; i < length; i++) {
                            GGUFMetadataValue elem(file, offset, elem_type);
                            vec.push_back(elem);
                        }
                        value = vec;
                        break;
                    }
                    default:
                        std::cout << "Unsupported GGUF metadata value type: " << (uint32_t)type << std::endl;
                        throw std::runtime_error("Unsupported GGUF metadata value type");
                }
            }

            std::string to_string() {
                switch (type) {
                    case GGUF_METADATA_VALUE_TYPE_UINT8:
                        return std::to_string(std::get<uint8_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_INT8:
                        return std::to_string(std::get<int8_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_UINT16:
                        return std::to_string(std::get<uint16_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_INT16:
                        return std::to_string(std::get<int16_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_UINT32:
                        return std::to_string(std::get<uint32_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_INT32:
                        return std::to_string(std::get<int32_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
                        return std::to_string(std::get<float>(value));
                    case GGUF_METADATA_VALUE_TYPE_BOOL:
                        return std::get<bool>(value) ? "true" : "false";
                    case GGUF_METADATA_VALUE_TYPE_STRING:
                        return std::get<std::string>(value);
                    case GGUF_METADATA_VALUE_TYPE_UINT64:
                        return std::to_string(std::get<uint64_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_INT64:
                        return std::to_string(std::get<int64_t>(value));
                    case GGUF_METADATA_VALUE_TYPE_FLOAT64:
                        return std::to_string(std::get<double>(value));
                    case GGUF_METADATA_VALUE_TYPE_ARRAY: {
                        std::string result = "[";
                        auto vec = std::get<std::vector<GGUFMetadataValue>>(value);

                        bool truncate = vec.size() > 10;
                        if (truncate) vec.resize(10);

                        for (size_t i = 0; i < vec.size(); i++) {
                            result += vec[i].to_string();
                            if (i + 1 < vec.size()) result += ", ";
                        }
                        
                        if (truncate) result += ", ...";
                        result += "]";
                        return result;
                    }
                    default:
                        return "Unsupported GGUF metadata value type";
                }
            }
        };

        struct tensor_data {
            ggml_type type;
            std::vector<uint32_t> dims;
            int offset;
            void* data;
        };

        class GGUFModel {
        public:
            MemoryMappedFile file;
            int version;
            uint64_t tensor_count;
            uint64_t kv_count;
            uint64_t current_offset = 24; // after header
            uint64_t alignment = 32;

            tokenizer::Tokenizer tk;

            std::vector<std::pair<std::string, GGUFMetadataValue>> metadata;
            std::vector<std::pair<std::string, tensor_data>> tensors;

            uint64_t align_offset(uint64_t offset) {
                return (offset + alignment - 1) & ~(alignment - 1);
            }

            void read_metadata_kv() {
                uint64_t length = file.read_at<uint64_t>(current_offset);
                current_offset += 8;
                std::string key = file.read_string_at(current_offset, length);
                current_offset += length;

                metadata.push_back({key, GGUFMetadataValue(file, current_offset)});
            }

            void load_metadata() {
                metadata.reserve(kv_count);
                for (uint64_t i = 0; i < kv_count; i++) {
                    read_metadata_kv();
                    // std::cout << "Metadata key: " << metadata.back().first 
                    //         << ", value: " << metadata.back().second.to_string() << std::endl;  
                }
            }

            void read_tensor() {
                uint64_t name_length = file.read_at<uint64_t>(current_offset);
                current_offset += 8;
                std::string name = file.read_string_at(current_offset, name_length);
                current_offset += name_length;

                uint32_t n_dims = file.read_at<uint32_t>(current_offset);
                current_offset += 4;
                std::vector<uint32_t> dims(n_dims);

                for (uint32_t i = 0; i < n_dims; i++) {
                    dims[i] = file.read_at<uint64_t>(current_offset);
                    current_offset += 8;
                }

                ggml_type type = file.read_at<ggml_type>(current_offset);
                current_offset += 4;
                // std::cout << "Tensor name: " << name << ", dims: [";
                // for (uint32_t i = 0; i < n_dims; i++) {
                //     std::cout << dims[i];
                //     if (i + 1 < n_dims) std::cout << ", ";
                // }
                // std::cout << "], type: " << (uint32_t)type << std::endl;

                uint64_t offset = file.read_at<uint64_t>(current_offset);
                current_offset += 8;

                // std::cout << "Tensor data offset: " << offset << std::endl;

                tensor_data tdata;
                tdata.type = type;
                
                std::reverse(dims.begin(), dims.end());
                tdata.dims = dims;
                
                tdata.offset = offset;
                tdata.data = nullptr;
                tensors.push_back({name, tdata});
            }

            void load_tensor() {
                for (uint64_t i = 0; i < tensor_count; i++)
                    read_tensor();
            }

            GGUFModel(const char* filepath) : file(filepath) {
                if (file.read_string_at(0, 4) != "GGUF")
                    throw std::runtime_error("Not a GGUF model");

                version = file.read_at<int32_t>(4);
                tensor_count = file.read_at<uint64_t>(8);
                kv_count = file.read_at<uint64_t>(16);

                std::cout << "Loaded GGUF model version " << version 
                        << " with " << tensor_count << " tensors and " 
                        << kv_count << " key-value pairs." << std::endl;

                load_metadata();
                load_tensor();

                current_offset = align_offset(current_offset);

                for (auto &[name, data] : tensors) {
                    assert(data.offset % alignment == 0 && "Tensor data offset is not aligned properly");
                    data.data = (char*)file.data + data.offset + current_offset;
                }
                
                std::vector<std::string> tokens;
                std::vector<std::pair<std::string, std::string>> merges;
                std::vector<int> token_type;
                
                for (auto &meta : metadata) {
                    if (meta.first == "tokenizer.ggml.tokens") {
                        auto& tk_data = std::get<std::vector<GGUFMetadataValue>>(meta.second.value);
                        tokens.reserve(tk_data.size());
                        for (auto &t : tk_data) 
                            tokens.push_back(std::get<std::string>(t.value));
                    }
                    else if (meta.first == "tokenizer.ggml.merges") {
                        auto& mk_data = std::get<std::vector<GGUFMetadataValue>>(meta.second.value);
                        merges.reserve(mk_data.size());
                        for (auto &m : mk_data) {
                            std::string merge_str = std::get<std::string>(m.value);
                            size_t space_pos = merge_str.find(' ');
                            if (space_pos != std::string::npos) {
                                merges.push_back({merge_str.substr(0, space_pos), merge_str.substr(space_pos + 1)});
                            }
                        }
                    }
                    else if (meta.first == "tokenizer.ggml.token_type") {
                        auto& tt_data = std::get<std::vector<GGUFMetadataValue>>(meta.second.value);
                        token_type.reserve(tt_data.size());
                        for (auto &tt : tt_data) {
                            token_type.push_back(std::get<int32_t>(tt.value));
                        }
                    }
                }

                tk = tokenizer::Tokenizer(tokens, merges, token_type);
            }
        };
    }
};