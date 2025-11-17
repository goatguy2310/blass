#include <cassert>
#include <functional>

inline std::string to_string_vec(const std::vector<size_t>& vec) {
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
    
    template <char op, typename T>
    void blass::elementwise_op(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& result, 
                               const std::vector<size_t>& shape, size_t dim, size_t offset_a, size_t offset_b, size_t offset_res, bool use_omp) {
        if (dim == shape.size() - 1 || result.strides[dim] == 1) {
            T* __restrict__ ptr_a = a.data.get() + offset_a;
            T* __restrict__ ptr_b = b.data.get() + offset_b;
            T* __restrict__ ptr_res = result.data.get() + offset_res;
            assert(a.strides[dim] <= 1 && b.strides[dim] <= 1 && "Strides for innermost dimension should be 0 or 1?");

            bool a_stride = (a.strides[dim] == 1);
            bool b_stride = (b.strides[dim] == 1);

            bool use_omp_local = use_omp && shape[dim] >= 32;

            if (a_stride && b_stride) {
                #pragma omp parallel for if(use_omp_local)
                for (size_t i = 0; i < shape[dim]; i++) {
                    if constexpr (op == '+')
                        ptr_res[i] = ptr_a[i] + ptr_b[i];
                    else if constexpr (op == '-')
                        ptr_res[i] = ptr_a[i] - ptr_b[i];
                    else if constexpr (op == '*')
                        ptr_res[i] = ptr_a[i] * ptr_b[i];
                    else if constexpr (op == '/')
                        ptr_res[i] = ptr_a[i] / ptr_b[i];
                }
            }

            if (a_stride && !b_stride) {
                #pragma omp parallel for if(use_omp_local)
                for (size_t i = 0; i < shape[dim]; i++) {
                    if constexpr (op == '+')
                        ptr_res[i] = ptr_a[i] + ptr_b[0];
                    else if constexpr (op == '-')
                        ptr_res[i] = ptr_a[i] - ptr_b[0];
                    else if constexpr (op == '*')
                        ptr_res[i] = ptr_a[i] * ptr_b[0];
                    else if constexpr (op == '/')
                        ptr_res[i] = ptr_a[i] / ptr_b[0];
                }
            }

            if (!a_stride && b_stride) {
                #pragma omp parallel for if(use_omp_local)
                for (size_t i = 0; i < shape[dim]; i++) {
                    if constexpr (op == '+')
                        ptr_res[i] = ptr_a[0] + ptr_b[i];
                    else if constexpr (op == '-')
                        ptr_res[i] = ptr_a[0] - ptr_b[i];
                    else if constexpr (op == '*')
                        ptr_res[i] = ptr_a[0] * ptr_b[i];
                    else if constexpr (op == '/')
                        ptr_res[i] = ptr_a[0] / ptr_b[i];
                }
            }

            if (!a_stride && !b_stride) {
                #pragma omp parallel for if(use_omp_local)
                for (size_t i = 0; i < shape[dim]; i++) {
                    if constexpr (op == '+')
                        ptr_res[i] = ptr_a[0] + ptr_b[0];
                    else if constexpr (op == '-')
                        ptr_res[i] = ptr_a[0] - ptr_b[0];
                    else if constexpr (op == '*')
                        ptr_res[i] = ptr_a[0] * ptr_b[0];
                    else if constexpr (op == '/')
                        ptr_res[i] = ptr_a[0] / ptr_b[0];
                }
            }
            return;
        }

        bool use_omp_local = use_omp && (result.strides.back() / result.strides[dim] <= 32 && shape[dim] >= 32);
        if (use_omp_local) use_omp = 0;

        #pragma omp parallel for if(use_omp_local)
        for (size_t i = 0; i < shape[dim]; i++) {
            elementwise_op<op>(a, b, result, shape, dim + 1, offset_a + i * a.strides[dim], offset_b + i * b.strides[dim], offset_res + i * result.strides[dim], use_omp);
        }
    }

    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        if (!a.is_contiguous() || !b.is_contiguous()) {
            return a.contiguous() + b.contiguous();
        }

        if (a.get_shape() == b.get_shape() && a.size() == b.size()) {
            Tensor<T> result = Tensor<T>::from_shape(a.get_shape());
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op<'+'> (a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0);

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
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op<'-'> (a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0);

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
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] * b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op<'*'>(a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0);

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
            
            #pragma omp parallel for if (a.size() >= 1024)
            for (size_t i = 0; i < a.size(); ++i) {
                result.data[i] = a.data[i] / b.data[i];
            }
            return result;
        }
        else {
            std::vector<size_t> result_shape = broadcast_shape(a.get_shape(), b.get_shape());
            Tensor<T> a_broadcasted = a.broadcast(result_shape);
            Tensor<T> b_broadcasted = b.broadcast(result_shape);
            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            elementwise_op<'/'>(a_broadcasted, b_broadcasted, result, result_shape, 0, 0, 0, 0);

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
    Tensor<T> matmul_2d(const Tensor<T> &a, const Tensor<T> &b, bool use_omp) {
        assert(a.get_shape().size() == 2 && b.get_shape().size() == 2 && "Both tensors must be 2D for matmul_2d");
        assert(a.get_shape(1) == b.get_shape(0) && "Inner dimensions must match for matmul_2d");

        size_t m = a.get_shape(0);
        size_t n = a.get_shape(1);
        size_t p = b.get_shape(1);

        Tensor<T> result = Tensor<T>::from_shape({m, p});

        Tensor<T> b_transposed = b.transpose();

        if (m >= p) {
            #pragma omp parallel for if (use_omp)
            for (size_t i = 0; i < m; ++i) {
                T* __restrict__ ptr_res_row = result.data.get() + i * result.strides[0];
                T* __restrict__ ptr_a_row = a.data.get() + i * a.strides[0];

                for (size_t j = 0; j < p; ++j) {
                    T* __restrict__ ptr_b_row = b_transposed.data.get() + j * b_transposed.strides[0];

                    T sum = 0;

                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < n; ++k)
                        sum += ptr_a_row[k] * ptr_b_row[k];

                    ptr_res_row[j] = sum;
                }
            }
        } 
        else {
            #pragma omp parallel for if (use_omp)
            for (size_t j = 0; j < p; ++j) {
                T* __restrict__ ptr_res_col = result.data.get() + j;
                T* __restrict__ ptr_b_row = b_transposed.data.get() + j * b_transposed.strides[0];
                
                for (size_t i = 0; i < m; ++i) {
                    T* __restrict__ ptr_a_row = a.data.get() + i * a.strides[0];

                    T sum = 0;

                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < n; ++k)
                        sum += ptr_a_row[k] * ptr_b_row[k];

                    ptr_res_col[i * result.strides[0]] = sum;
                }
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b) {
        if (a.shape.size() < 2 || b.shape.size() < 2) {
            throw std::invalid_argument("Both tensors must be at least 2D for matmul");
        }

        if (a.shape.size() == 2 && b.shape.size() == 2) {
            return matmul_2d(a, b);
        }
        else {
            std::vector<size_t> batch_a_shape, batch_b_shape;
            for (size_t i = 0; i < a.shape.size() - 2; i++)
                batch_a_shape.push_back(a.shape[i]);

            for (size_t i = 0; i < b.shape.size() - 2; i++)
                batch_b_shape.push_back(b.shape[i]);

            if (batch_a_shape.size() == 0)
                batch_a_shape.push_back(1);
            
            if (batch_b_shape.size() == 0)
                batch_b_shape.push_back(1);
            
            std::vector<size_t> batch_result_shape = broadcast_shape(batch_a_shape, batch_b_shape);
            if (a.shape[a.shape.size() - 1] != b.shape[b.shape.size() - 2]) {
                throw std::invalid_argument("Inner dimensions must match for matmul");
            }

            std::vector<size_t> a_broadcast_shape = batch_result_shape;
            a_broadcast_shape.push_back(a.shape[a.shape.size() - 2]);
            a_broadcast_shape.push_back(a.shape[a.shape.size() - 1]);

            std::vector<size_t> b_broadcast_shape = batch_result_shape;
            b_broadcast_shape.push_back(b.shape[b.shape.size() - 2]);
            b_broadcast_shape.push_back(b.shape[b.shape.size() - 1]);

            Tensor<T> a_broadcasted = a.broadcast(a_broadcast_shape);
            Tensor<T> b_broadcasted = b.broadcast(b_broadcast_shape);

            std::vector<size_t> result_shape = batch_result_shape;
            result_shape.push_back(a.shape[a.shape.size() - 2]);
            result_shape.push_back(b.shape[b.shape.size() - 1]);

            Tensor<T> result = Tensor<T>::from_shape(result_shape);

            size_t batch_size = 1;

            for (size_t dim : batch_result_shape)
                batch_size *= dim;

            std::vector<size_t> idx(batch_result_shape.size(), 0);
            
            const auto& sa = a_broadcasted.strides;
            const auto& sb = b_broadcasted.strides;

            bool use_omp_batch = batch_size >= 32;

            #pragma omp parallel for if(use_omp_batch)
            for (size_t i = 0; i < batch_size; ++i) {
                size_t offset_a = 0, offset_b = 0, offset_res = 0;
                for (size_t dim = 0; dim < batch_result_shape.size(); ++dim) {
                    offset_a += idx[dim] * sa[dim];
                    offset_b += idx[dim] * sb[dim];
                    offset_res += idx[dim] * result.strides[dim];
                }

                std::shared_ptr<T[]> ptr_a = std::shared_ptr<T[]>(a_broadcasted.data, a_broadcasted.data.get() + offset_a);
                std::shared_ptr<T[]> ptr_b = std::shared_ptr<T[]>(b_broadcasted.data, b_broadcasted.data.get() + offset_b);
                std::shared_ptr<T[]> ptr_res = std::shared_ptr<T[]>(result.data, result.data.get() + offset_res);

                Tensor<T> a_slice(ptr_a, 
                                {a.shape[a.shape.size() - 2], a.shape[a.shape.size() - 1]});
                Tensor<T> b_slice(ptr_b, 
                                 {b.shape[b.shape.size() - 2], b.shape[b.shape.size() - 1]});
                Tensor<T> result_slice(ptr_res, 
                                      {result_shape[result_shape.size() - 2], result_shape[result_shape.size() - 1]});

                Tensor<T> mat_result = matmul_2d(a_slice, b_slice, !use_omp_batch);
                
                std::cout << "A slice value: " << a_slice.to_string() << "\n";
                std::cout << "B slice value: " << b_slice.to_string() << "\n";

                std::cout << "Mat result shape: " << to_string_vec(mat_result.get_shape()) << "\n";
                std::cout << "Mat result value:" << mat_result.to_string() << "\n";
                std::copy(mat_result.data.get(), mat_result.data.get() + mat_result.size(), result_slice.data.get());

                for (int dim = batch_result_shape.size() - 1; dim >= 0; --dim) {
                    idx[dim]++;
                    if (idx[dim] < batch_result_shape[dim])
                        break;
                    idx[dim] = 0;
                }
            }
            return result;
        }
    }
}