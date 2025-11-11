#include <cassert>

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
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for addition");
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() + b.contiguous();
        }
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data[i] = a.data[i] + b.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> subtract(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for subtraction");
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() - b.contiguous();
        }
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data[i] = a.data[i] - b.data[i];
        }
        return result;
    }

    template <typename T>
    Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.get_shape() == b.get_shape() && "Shapes must match for multiplication");
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() * b.contiguous();
        }
        Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
        for (size_t i = 0; i < a.size(); ++i) {
            result.data[i] = a.data[i] * b.data[i];
        }
        return result;
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