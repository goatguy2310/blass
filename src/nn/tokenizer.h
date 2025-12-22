#pragma once

#include <vector>
#include <map>
#include <codecvt>

namespace blass {
    namespace tokenizer {
        class Tokenizer {
            std::vector<std::string> tokens;
            std::map<std::string, int> encoder;
            std::vector<std::pair<std::string, std::string>> merges;
            std::vector<int> token_type;
        public:
            Tokenizer() {}
            Tokenizer(std::vector<std::string> tokens_, std::vector<std::pair<std::string, std::string>> merges_, std::vector<int> token_type_) : tokens(tokens_), merges(merges_), token_type(token_type_) {
                for (size_t i = 0; i < tokens.size(); i++)
                    encoder[tokens[i]] = i;
            }

            // Removed in c++26 but theres no replacement yet so ima just leave it here
            std::u32string utf8_to_u32(const std::string &s) {
                std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
                return conv.from_bytes(s);
            }

            std::string u32_to_utf8(char32_t c) {
                std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
                return conv.to_bytes(c);
            }

            std::vector<int> encode(const std::string& _text) {
                std::u32string text = utf8_to_u32(_text);
                std::vector<std::string> words;
                words.reserve(text.size());
                for (char32_t c : text) {
                    //"Ġ"
                    if (c == ' ') c = U'Ġ';
                    words.push_back(u32_to_utf8(c));
                }

                for (auto &[first, second]: merges) {
                    std::vector<std::string> new_words;
                    for (size_t i = 0; i < words.size(); i++) {
                        if (i + 1 < words.size() && words[i] == first && words[i + 1] == second) {
                            new_words.push_back(first + second);
                            i++;
                        }
                        else new_words.push_back(words[i]);
                    }

                    words = std::move(new_words);
                }

                for (auto &i: words)
                    std::cout << i << ' '; 
                std::cout << std::endl;

                std::vector<int> tokens;
                for (auto &i: words)
                    tokens.push_back(encoder[i]);
                return tokens;
            }

            std::string decode(const std::vector<int>& token_ids) {
                return "";
            }
        };
    }
};