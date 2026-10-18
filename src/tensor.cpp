// Tensor implementation: storage, indexing, broadcasting, matmul, reductions.
#include "nn/tensor.hpp"

#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace nn {
namespace {

std::mt19937& rng() {
    static std::mt19937 gen(42);
    return gen;
}

int64_t offset_of(const std::vector<int64_t>& strides, const std::vector<int64_t>& idx) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

// Increment a multi-index over `shape` (row-major). Returns false on wrap.
bool bump(std::vector<int64_t>& idx, const std::vector<int64_t>& shape) {
    for (int64_t i = static_cast<int64_t>(idx.size()) - 1; i >= 0; --i) {
        if (++idx[i] < shape[i]) return true;
        idx[i] = 0;
    }
    return false;
}

Tensor binary_broadcast(const Tensor& a, const Tensor& b, float (*f)(float, float)) {
    auto shape = broadcast_shapes(a.shape(), b.shape());
    Tensor out(shape);
    std::vector<int64_t> idx(shape.size(), 0);
    std::vector<int64_t> ia(a.ndim()), ib(b.ndim());
    const int64_t off = static_cast<int64_t>(shape.size());
    do {
        // Map output index onto each operand (dim of size 1 -> index 0).
        for (int64_t d = 0; d < a.ndim(); ++d)
            ia[d] = (a.dim(d) == 1) ? 0 : idx[off - a.ndim() + d];
        for (int64_t d = 0; d < b.ndim(); ++d)
            ib[d] = (b.dim(d) == 1) ? 0 : idx[off - b.ndim() + d];
        out[offset_of(out.strides_, idx)] =
            f(a.data_[offset_of(a.strides_, ia)], b.data_[offset_of(b.strides_, ib)]);
    } while (bump(idx, shape));
    return out;
}

float f_add(float x, float y) { return x + y; }
float f_sub(float x, float y) { return x - y; }
float f_mul(float x, float y) { return x * y; }
float f_div(float x, float y) { return x / y; }

Tensor unary(const Tensor& a, float (*f)(float)) {
    Tensor out(a.shape());
    for (int64_t i = 0; i < a.numel(); ++i) out[i] = f(a[i]);
    return out;
}

float f_exp(float x) { return std::exp(x); }
float f_log(float x) { return std::log(x > 1e-12f ? x : 1e-12f); }
float f_sqrt(float x) { return std::sqrt(x > 0.0f ? x : 0.0f); }
float f_sq(float x) { return x * x; }
float f_neg(float x) { return -x; }
float f_relu(float x) { return x > 0.0f ? x : 0.0f; }
float f_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) {
        if (d < 0) throw std::invalid_argument("negative dimension");
        n *= d;
    }
    return n;
}

std::vector<int64_t> row_major_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= shape[i];
    }
    return s;
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                      const std::vector<int64_t>& b) {
    const size_t n = std::max(a.size(), b.size());
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        const int64_t da = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
        const int64_t db = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("shapes are not broadcastable");
        out[i] = std::max(da, db);
    }
    return out;
}

Tensor::Tensor(std::vector<int64_t> shape)
    : shape_(std::move(shape)), strides_(row_major_strides(shape_)),
      data_(static_cast<size_t>(numel_of(shape_)), 0.0f) {}

Tensor::Tensor(std::vector<int64_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), strides_(row_major_strides(shape_)), data_(std::move(data)) {
    if (static_cast<int64_t>(data_.size()) != numel_of(shape_))
        throw std::invalid_argument("data size does not match shape");
}

Tensor Tensor::zeros(std::vector<int64_t> shape) { return Tensor(std::move(shape)); }

Tensor Tensor::ones(std::vector<int64_t> shape) {
    return full(std::move(shape), 1.0f);
}

Tensor Tensor::full(std::vector<int64_t> shape, float v) {
    Tensor t(std::move(shape));
    std::fill(t.data_.begin(), t.data_.end(), v);
    return t;
}

Tensor Tensor::arange(int64_t n) {
    Tensor t({n});
    std::iota(t.data_.begin(), t.data_.end(), 0.0f);
    return t;
}

Tensor Tensor::randn(std::vector<int64_t> shape, float mean, float std) {
    Tensor t(std::move(shape));
    std::normal_distribution<float> d(mean, std);
    for (auto& v : t.data_) v = d(rng());
    return t;
}

Tensor Tensor::randu(std::vector<int64_t> shape, float lo, float hi) {
    Tensor t(std::move(shape));
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : t.data_) v = d(rng());
    return t;
}

int64_t Tensor::dim(int64_t i) const {
    if (i < 0) i += ndim();
    if (i < 0 || i >= ndim()) throw std::out_of_range("dim index out of range");
    return shape_[static_cast<size_t>(i)];
}

float& Tensor::at(std::vector<int64_t> idx) {
    if (static_cast<int64_t>(idx.size()) != ndim())
        throw std::invalid_argument("index rank mismatch");
    return data_[static_cast<size_t>(offset_of(strides_, idx))];
}

float Tensor::at(std::vector<int64_t> idx) const {
    if (static_cast<int64_t>(idx.size()) != ndim())
        throw std::invalid_argument("index rank mismatch");
    return data_[static_cast<size_t>(offset_of(strides_, idx))];
}

Tensor Tensor::reshape(std::vector<int64_t> shape) const {
    int64_t known = 1;
    int64_t infer = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer != -1) throw std::invalid_argument("only one -1 allowed in reshape");
            infer = static_cast<int64_t>(i);
        } else {
            known *= shape[i];
        }
    }
    if (infer != -1) {
        if (numel() % known != 0) throw std::invalid_argument("reshape size mismatch");
        shape[static_cast<size_t>(infer)] = numel() / known;
    }
    if (numel_of(shape) != numel()) throw std::invalid_argument("reshape size mismatch");
    return Tensor(std::move(shape), data_); // shares values (copy here — simple + safe)
}

Tensor Tensor::transpose2d() const {
    if (ndim() != 2) throw std::invalid_argument("transpose2d needs a 2-D tensor");
    const int64_t r = dim(0), c = dim(1);
    Tensor out({c, r});
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < c; ++j) out.at({j, i}) = at({i, j});
    return out;
}

Tensor Tensor::broadcast_to(std::vector<int64_t> shape) const {
    (void)broadcast_shapes(shape_, shape); // validates
    Tensor out(shape);
    std::vector<int64_t> idx(shape.size(), 0);
    std::vector<int64_t> isrc(ndim());
    const int64_t off = static_cast<int64_t>(shape.size());
    do {
        for (int64_t d = 0; d < ndim(); ++d)
            isrc[d] = (dim(d) == 1) ? 0 : idx[off - ndim() + d];
        out[offset_of(out.strides_, idx)] = data_[offset_of(strides_, isrc)];
    } while (bump(idx, shape));
    return out;
}

Tensor Tensor::slice_rows(int64_t start, int64_t end) const {
    if (ndim() != 2) throw std::invalid_argument("slice_rows needs a 2-D tensor");
    if (start < 0 || end > dim(0) || start > end)
        throw std::out_of_range("slice_rows range out of bounds");
    const int64_t c = dim(1);
    Tensor out({end - start, c});
    for (int64_t i = start; i < end; ++i)
        for (int64_t j = 0; j < c; ++j) out.at({i - start, j}) = at({i, j});
    return out;
}

std::string Tensor::to_string(int max_elems) const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) os << ", ";
        os << shape_[i];
    }
    os << "], data=[";
    const int64_t n = std::min<int64_t>(numel(), max_elems);
    for (int64_t i = 0; i < n; ++i) {
        if (i) os << ", ";
        os << data_[static_cast<size_t>(i)];
    }
    if (numel() > max_elems) os << ", ...";
    os << "])";
    return os.str();
}

Tensor add(const Tensor& a, const Tensor& b) { return binary_broadcast(a, b, f_add); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary_broadcast(a, b, f_sub); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary_broadcast(a, b, f_mul); }
Tensor div(const Tensor& a, const Tensor& b) { return binary_broadcast(a, b, f_div); }
Tensor neg(const Tensor& a) { return unary(a, f_neg); }
Tensor add_scalar(const Tensor& a, float s) { return add(a, Tensor::full({}, s)); }
Tensor mul_scalar(const Tensor& a, float s) { return mul(a, Tensor::full({}, s)); }

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::invalid_argument("matmul needs 2-D tensors");
    const int64_t m = a.dim(0), k = a.dim(1), n = b.dim(1);
    if (b.dim(0) != k) throw std::invalid_argument("matmul inner dimensions mismatch");
    Tensor out({m, n});
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t kk = 0; kk < k; ++kk) {
            const float aik = a.at({i, kk});
            for (int64_t j = 0; j < n; ++j) out.at({i, j}) += aik * b.at({kk, j});
        }
    }
    return out;
}

Tensor sum_all(const Tensor& a) {
    float s = 0.0f;
    for (float v : a.data_) s += v;
    return Tensor::full({}, s);
}

Tensor mean_all(const Tensor& a) {
    return Tensor::full({}, sum_all(a).at({}) / static_cast<float>(a.numel()));
}

Tensor sum_rows(const Tensor& a) {
    if (a.ndim() != 2) throw std::invalid_argument("sum_rows needs a 2-D tensor");
    Tensor out({a.dim(1)});
    for (int64_t i = 0; i < a.dim(0); ++i)
        for (int64_t j = 0; j < a.dim(1); ++j) out[j] += a.at({i, j});
    return out;
}

Tensor sum_cols(const Tensor& a) {
    if (a.ndim() != 2) throw std::invalid_argument("sum_cols needs a 2-D tensor");
    Tensor out({a.dim(0)});
    for (int64_t i = 0; i < a.dim(0); ++i)
        for (int64_t j = 0; j < a.dim(1); ++j) out[i] += a.at({i, j});
    return out;
}

Tensor exp_t(const Tensor& a) { return unary(a, f_exp); }
Tensor log_t(const Tensor& a) { return unary(a, f_log); }
Tensor sqrt_t(const Tensor& a) { return unary(a, f_sqrt); }
Tensor pow2(const Tensor& a) { return unary(a, f_sq); }
Tensor relu_fwd(const Tensor& a) { return unary(a, f_relu); }
Tensor sigmoid_fwd(const Tensor& a) { return unary(a, f_sigmoid); }

Tensor tanh_fwd(const Tensor& a) {
    return unary(a, [](float x) { return std::tanh(x); });
}

Tensor clamp_min(const Tensor& a, float v) {
    return unary(a, [v](float x) { return x < v ? v : x; });
}

Tensor softmax_rows(const Tensor& a) {
    if (a.ndim() != 2) throw std::invalid_argument("softmax_rows needs a 2-D tensor");
    Tensor out(a.shape());
    const int64_t n = a.dim(0), c = a.dim(1);
    for (int64_t i = 0; i < n; ++i) {
        float mx = a.at({i, 0});
        for (int64_t j = 1; j < c; ++j) mx = std::max(mx, a.at({i, j}));
        float s = 0.0f;
        for (int64_t j = 0; j < c; ++j) {
            const float e = std::exp(a.at({i, j}) - mx);
            out.at({i, j}) = e;
            s += e;
        }
        for (int64_t j = 0; j < c; ++j) out.at({i, j}) /= s;
    }
    return out;
}

Tensor log_softmax_rows(const Tensor& a) {
    if (a.ndim() != 2) throw std::invalid_argument("log_softmax_rows needs a 2-D tensor");
    Tensor out(a.shape());
    const int64_t n = a.dim(0), c = a.dim(1);
    for (int64_t i = 0; i < n; ++i) {
        float mx = a.at({i, 0});
        for (int64_t j = 1; j < c; ++j) mx = std::max(mx, a.at({i, j}));
        float s = 0.0f;
        for (int64_t j = 0; j < c; ++j) s += std::exp(a.at({i, j}) - mx);
        const float lse = std::log(s);
        for (int64_t j = 0; j < c; ++j) out.at({i, j}) = a.at({i, j}) - mx - lse;
    }
    return out;
}

Tensor unbroadcast(const Tensor& grad, const std::vector<int64_t>& shape) {
    if (grad.shape() == shape) return grad;
    // Sum out leading extra dims, then dims where target has size 1.
    Tensor g = grad;
    while (static_cast<int64_t>(g.ndim()) > static_cast<int64_t>(shape.size())) {
        // sum over dim 0
        if (g.ndim() == 2) {
            g = sum_rows(g).reshape({1, g.dim(1)});
        } else if (g.ndim() == 1) {
            g = sum_all(g); // scalar
        } else {
            throw std::invalid_argument("unbroadcast: unsupported rank");
        }
    }
    auto cur = g.shape();
    for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] == 1 && cur[d] != 1) {
            if (cur.size() == 2 && d == 0) {
                g = sum_rows(g).reshape({1, cur[1]});
            } else if (cur.size() == 2 && d == 1) {
                g = sum_cols(g).reshape({cur[0], 1});
            } else if (cur.size() == 1) {
                g = sum_all(g);
            } else {
                throw std::invalid_argument("unbroadcast: unsupported rank");
            }
            cur = g.shape();
        }
    }
    return g.reshape(shape);
}

int64_t argmax_row(const Tensor& a, int64_t row) {
    if (a.ndim() != 2) throw std::invalid_argument("argmax_row needs a 2-D tensor");
    int64_t best = 0;
    float bv = a.at({row, 0});
    for (int64_t j = 1; j < a.dim(1); ++j) {
        const float v = a.at({row, j});
        if (v > bv) {
            bv = v;
            best = j;
        }
    }
    return best;
}

void set_seed(uint64_t s) { rng().seed(static_cast<std::mt19937::result_type>(s)); }

} // namespace nn
