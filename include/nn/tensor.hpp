// Tensor: N-dimensional dense array, row-major, float32.
// No dependencies — every kernel below is written by hand.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nn {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape);                 // zero-filled
    Tensor(std::vector<int64_t> shape, std::vector<float> data); // data must match numel

    static Tensor zeros(std::vector<int64_t> shape);
    static Tensor ones(std::vector<int64_t> shape);
    static Tensor full(std::vector<int64_t> shape, float v);
    static Tensor arange(int64_t n);
    static Tensor randn(std::vector<int64_t> shape, float mean = 0.0f, float std = 1.0f);
    static Tensor randu(std::vector<int64_t> shape, float lo = 0.0f, float hi = 1.0f);

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return static_cast<int64_t>(data_.size()); }
    int64_t dim(int64_t i) const;

    float& operator[](int64_t i) { return data_[i]; }
    float operator[](int64_t i) const { return data_[i]; }
    float& at(std::vector<int64_t> idx);
    float at(std::vector<int64_t> idx) const;

    bool same_shape(const Tensor& o) const { return shape_ == o.shape_; }

    Tensor reshape(std::vector<int64_t> shape) const;
    Tensor transpose2d() const;                        // 2-D only
    Tensor broadcast_to(std::vector<int64_t> shape) const;
    Tensor slice_rows(int64_t start, int64_t end) const; // 2-D row slice [start, end)

    std::string to_string(int max_elems = 12) const;

    // Public for simplicity: this is a teaching framework, not a vault.
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_; // row-major, contiguous
    std::vector<float> data_;
};

int64_t numel_of(const std::vector<int64_t>& shape);
std::vector<int64_t> row_major_strides(const std::vector<int64_t>& shape);
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                      const std::vector<int64_t>& b);

// Elementwise (NumPy-style broadcasting).
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);
Tensor add_scalar(const Tensor& a, float s);
Tensor mul_scalar(const Tensor& a, float s);

// Linear algebra.
Tensor matmul(const Tensor& a, const Tensor& b); // 2-D @ 2-D

// Reductions.
Tensor sum_all(const Tensor& a);   // scalar tensor, shape {}
Tensor mean_all(const Tensor& a);  // scalar tensor, shape {}
Tensor sum_rows(const Tensor& a);  // [N, C] -> [C]
Tensor sum_cols(const Tensor& a);  // [N, C] -> [N]

// Unary math.
Tensor exp_t(const Tensor& a);
Tensor log_t(const Tensor& a); // log(max(x, 1e-12)) for stability
Tensor sqrt_t(const Tensor& a);
Tensor pow2(const Tensor& a);
Tensor relu_fwd(const Tensor& a);
Tensor sigmoid_fwd(const Tensor& a);
Tensor tanh_fwd(const Tensor& a);
Tensor clamp_min(const Tensor& a, float v);

// Row-wise normalisation for [N, C] tensors.
Tensor softmax_rows(const Tensor& a);
Tensor log_softmax_rows(const Tensor& a);

// Collapse a broadcast gradient back to a parameter's shape.
Tensor unbroadcast(const Tensor& grad, const std::vector<int64_t>& shape);

int64_t argmax_row(const Tensor& a, int64_t row); // 2-D

// Global RNG (seeded for reproducible experiments).
void set_seed(uint64_t s);

} // namespace nn
