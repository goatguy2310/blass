#include <cassert>
#include <functional>

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
        for (size_t i = 0; i < sz; i++) {
            size_t idx = i;
            std::vector<size_t> multi_idx(shape.size());
            for (size_t d = 0; d < shape.size(); d++) {
                multi_idx[d] = idx / strides[d];
                idx = idx % strides[d];
            }

            size_t orig_offset = 0;
            for (size_t d = 0; d < shape.size(); d++) {
                orig_offset += multi_idx[d] * strides[d];
            }
            result.data[i] = data[orig_offset];
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
        return result;
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
    std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b) {
        size_t len_a = shape_a.size();
        size_t len_b = shape_b.size();
        size_t len_result = std::max(len_a, len_b);
        std::vector<size_t> result_shape(len_result, 1);

        for (size_t i = 0; i < len_result; i++) {
            size_t dim_a = (i < len_result - len_a) ? 1 : shape_a[i - (len_result - len_a)];
            size_t dim_b = (i < len_result - len_b) ? 1 : shape_b[i - (len_result - len_b)];

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::invalid_argument("Shapes cannot be broadcasted: " + to_string_vec(shape_a) + " and " + to_string_vec(shape_b));
            }
            result_shape[i] = std::max(dim_a, dim_b);
        }
        return result_shape;
    }

    template <typename T>
    Tensor<T> Tensor<T>::broadcast(const std::vector<size_t>& target_shape) const {
        std::vector<size_t> broadcasted_stride(target_shape.size(), 0);
        if (target_shape.size() < shape.size()) {
            throw std::invalid_argument("Cannot broadcast tensor of shape " + to_string_vec(shape) + " to target shape " + to_string_vec(target_shape));
        }
        size_t offset = target_shape.size() - shape.size();
        for (size_t i = 0; i < target_shape.size(); i++) {
            size_t current_dim = (i < offset) ? 1 : shape[i - offset];

            if (current_dim == target_shape[i]) 
                broadcasted_stride[i] = (i < offset) ? 0 : strides[i - offset];
            else if (current_dim == 1) 
                broadcasted_stride[i] = 0;
            else 
                throw std::invalid_argument("Cannot broadcast tensor of shape " + to_string_vec(shape) + " to target shape " + to_string_vec(target_shape));
        }
        return Tensor<T>(data, target_shape, broadcasted_stride);
    }
    
    template <typename T>
    void blass::elementwise_op(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& result, 
                               const std::vector<size_t>& shape, size_t dim, size_t offset_a, size_t offset_b, size_t offset_res, char op) {
        if (dim == shape.size() - 1) {
            for (size_t i = 0; i < shape[dim]; i++) {
                T* ptr_a = a.data.get();
                T* ptr_b = b.data.get();
                T* ptr_res = result.data.get();
                assert(a.strides[dim] <= 1 && b.strides[dim] <= 1 && "Strides for innermost dimension should be 0 or 1?");

                bool a_stride = (a.strides[dim] == 1);
                bool b_stride = (b.strides[dim] == 1);

                if (op == '+')
                    ptr_res[offset_res + i] = ptr_a[a_stride ? offset_a + i : offset_a] + ptr_b[b_stride ? offset_b + i : offset_b];
                else if (op == '-')
                    ptr_res[offset_res + i] = ptr_a[a_stride ? offset_a + i : offset_a] - ptr_b[b_stride ? offset_b + i : offset_b];
                else if (op == '*')
                    ptr_res[offset_res + i] = ptr_a[a_stride ? offset_a + i : offset_a] * ptr_b[b_stride ? offset_b + i : offset_b];
                else if (op == '/')
                    ptr_res[offset_res + i] = ptr_a[a_stride ? offset_a + i : offset_a] / ptr_b[b_stride ? offset_b + i : offset_b];
                else
                    throw std::invalid_argument("Unsupported operation in elementwise_op");
            }
            return;
        }
        for (size_t i = 0; i < shape[dim]; i++) 
            elementwise_op(a, b, result, shape, dim + 1, offset_a + i * a.strides[dim], offset_b + i * b.strides[dim], offset_res + i * result.strides[dim], op);
    }

    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() + b.contiguous();
        }

        if (a.get_shape() == b.get_shape() && a.size() == b.size()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape<T>(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op(a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0, '+');

            return result;
        }
    }

    template <typename T>
    Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() - b.contiguous();
        }

        if (a.get_shape() == b.get_shape() && a.size() == b.size()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape<T>(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op(a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0, '-');

            return result;
        }
    }

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() * b.contiguous();
        }

        if (a.get_shape() == b.get_shape() && a.size() == b.size()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] * b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape<T>(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op(a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0, '*');

            return result;
        }
    }

    template <typename T>
    Tensor<T> divide(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() / b.contiguous();
        }

        if (a.get_shape() == b.get_shape() && a.size() == b.size()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] / b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape<T>(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op(a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0, '/');

            return result;
        }
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
    Tensor<T> matmul_2d(const Tensor<T> &a, const Tensor<T> &b) {
        assert(a.get_shape().size() == 2 && b.get_shape().size() == 2 && "Both tensors must be 2D for matmul_2d");
        assert(a.get_shape(1) == b.get_shape(0) && "Inner dimensions must match for matmul_2d");

        size_t m = a.get_shape(0);
        size_t n = a.get_shape(1);
        size_t p = b.get_shape(1);

        Tensor<T> result = Tensor<T>::from_shape({m, p});

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                T sum = 0;
                for (size_t k = 0; k < n; ++k)
                    sum += a(i, k) * b(k, j);

                result(i, j) = sum;
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b) {
        throw std::runtime_error("Generalized matmul not implemented yet.");
    }
}