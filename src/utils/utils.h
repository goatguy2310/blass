#pragma once

#include <string>
#include <vector>
#include <sstream>

namespace utils {
std::string to_string_vec(const std::vector<size_t>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

template <char op, typename T>
T scalar_op(const T a, const T b) {
    if constexpr (op == '+')
        return a + b;
    else if constexpr (op == '-')
        return  a - b;
    else if constexpr (op == '*')
        return a * b;
    else if constexpr (op == '/')
        return a / b;

    throw std::invalid_argument("Unsupported operation");
}
}