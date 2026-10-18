// Autograd engine implementation.
#include "nn/autograd.hpp"

#include <stdexcept>
#include <unordered_set>

namespace nn {
namespace {

// Attach a node to `out` only if at least one input needs a gradient.
void maybe_record(Var& out, std::vector<Var> inputs,
                  std::function<void(const Tensor&)> backward_fn) {
    bool any = false;
    for (auto& i : inputs)
        if (i->requires_grad) {
            any = true;
            break;
        }
    if (!any) return;
    auto node = std::make_shared<Node>();
    node->inputs = std::move(inputs);
    node->backward_fn = std::move(backward_fn);
    out->creator = std::move(node);
}

void topo_sort(Var v, std::unordered_set<Variable*>& seen, std::vector<Var>& order) {
    if (!v || seen.count(v.get())) return;
    seen.insert(v.get());
    if (v->creator)
        for (auto& in : v->creator->inputs) topo_sort(in, seen, order);
    order.push_back(v);
}

} // namespace

Var make_var(Tensor t, bool requires_grad) {
    return std::make_shared<Variable>(std::move(t), requires_grad);
}

Var constant(Tensor t) { return make_var(std::move(t), false); }

void Variable::accumulate_grad(const Tensor& g) {
    if (!requires_grad) return;
    if (grad.numel() == 0)
        grad = g;
    else
        grad = add(grad, g);
}

void Variable::backward() {
    if (data.numel() != 1)
        throw std::invalid_argument("backward() needs a scalar output");
    std::unordered_set<Variable*> seen;
    std::vector<Var> order;
    topo_sort(shared_from_this(), seen, order);
    grad = Tensor::ones(data.shape()); // dL/dL = 1
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Var v = *it;
        if (v->creator && v->grad.numel() != 0) v->creator->backward_fn(v->grad);
    }
}

// --- elementwise ---------------------------------------------------------

Var add(Var a, Var b) {
    auto out = make_var(nn::add(a->data, b->data));
    maybe_record(out, {a, b}, [a, b](const Tensor& g) {
        a->accumulate_grad(unbroadcast(g, a->data.shape()));
        b->accumulate_grad(unbroadcast(g, b->data.shape()));
    });
    return out;
}

Var sub(Var a, Var b) {
    auto out = make_var(nn::sub(a->data, b->data));
    maybe_record(out, {a, b}, [a, b](const Tensor& g) {
        a->accumulate_grad(unbroadcast(g, a->data.shape()));
        b->accumulate_grad(unbroadcast(neg(g), b->data.shape()));
    });
    return out;
}

Var mul(Var a, Var b) {
    auto out = make_var(nn::mul(a->data, b->data));
    const Tensor ad = a->data, bd = b->data;
    maybe_record(out, {a, b}, [a, b, ad, bd](const Tensor& g) {
        a->accumulate_grad(unbroadcast(nn::mul(g, bd.broadcast_to(g.shape())), a->data.shape()));
        b->accumulate_grad(unbroadcast(nn::mul(g, ad.broadcast_to(g.shape())), b->data.shape()));
    });
    return out;
}

Var div(Var a, Var b) {
    auto out = make_var(nn::div(a->data, b->data));
    const Tensor ad = a->data, bd = b->data;
    maybe_record(out, {a, b}, [a, b, ad, bd](const Tensor& g) {
        const Tensor bs = bd.broadcast_to(g.shape());
        const Tensor as = ad.broadcast_to(g.shape());
        a->accumulate_grad(unbroadcast(nn::div(g, bs), a->data.shape()));
        // d(a/b)/db = -a/b^2
        Tensor neg_ab2 = nn::div(as, nn::mul(bs, bs));
        for (auto& v : neg_ab2.data_) v = -v;
        b->accumulate_grad(unbroadcast(nn::mul(g, neg_ab2), b->data.shape()));
    });
    return out;
}

Var neg(Var a) {
    auto out = make_var(nn::neg(a->data));
    maybe_record(out, {a}, [a](const Tensor& g) { a->accumulate_grad(nn::neg(g)); });
    return out;
}

Var add_scalar(Var a, float s) { return add(a, constant(Tensor::full({}, s))); }
Var mul_scalar(Var a, float s) { return mul(a, constant(Tensor::full({}, s))); }

// --- matmul --------------------------------------------------------------

Var matmul(Var a, Var b) {
    auto out = make_var(nn::matmul(a->data, b->data));
    const Tensor ad = a->data, bd = b->data;
    maybe_record(out, {a, b}, [a, b, ad, bd](const Tensor& g) {
        a->accumulate_grad(nn::matmul(g, bd.transpose2d()));
        b->accumulate_grad(nn::matmul(ad.transpose2d(), g));
    });
    return out;
}

// --- unary activations ---------------------------------------------------

Var relu(Var a) {
    auto out = make_var(relu_fwd(a->data));
    const Tensor od = out->data;
    maybe_record(out, {a}, [a, od](const Tensor& g) {
        Tensor ga = g;
        for (int64_t i = 0; i < ga.numel(); ++i)
            if (!(od[i] > 0.0f)) ga[i] = 0.0f;
        a->accumulate_grad(unbroadcast(ga, a->data.shape()));
    });
    return out;
}

Var sigmoid(Var a) {
    auto out = make_var(sigmoid_fwd(a->data));
    const Tensor od = out->data;
    maybe_record(out, {a}, [a, od](const Tensor& g) {
        Tensor ga = g;
        for (int64_t i = 0; i < ga.numel(); ++i) {
            const float s = od[i];
            ga[i] *= s * (1.0f - s);
        }
        a->accumulate_grad(unbroadcast(ga, a->data.shape()));
    });
    return out;
}

Var tanh_act(Var a) {
    auto out = make_var(tanh_fwd(a->data));
    const Tensor od = out->data;
    maybe_record(out, {a}, [a, od](const Tensor& g) {
        Tensor ga = g;
        for (int64_t i = 0; i < ga.numel(); ++i) ga[i] *= (1.0f - od[i] * od[i]);
        a->accumulate_grad(unbroadcast(ga, a->data.shape()));
    });
    return out;
}

Var exp_v(Var a) {
    auto out = make_var(exp_t(a->data));
    const Tensor od = out->data;
    maybe_record(out, {a}, [a, od](const Tensor& g) {
        a->accumulate_grad(unbroadcast(nn::mul(g, od), a->data.shape()));
    });
    return out;
}

Var log_v(Var a) {
    auto out = make_var(log_t(a->data));
    const Tensor ad = a->data;
    maybe_record(out, {a}, [a, ad](const Tensor& g) {
        Tensor inv = ad;
        for (auto& v : inv.data_) v = 1.0f / (v > 1e-12f ? v : 1e-12f);
        a->accumulate_grad(unbroadcast(nn::mul(g, inv), a->data.shape()));
    });
    return out;
}

Var pow2_v(Var a) {
    auto out = make_var(pow2(a->data));
    const Tensor ad = a->data;
    maybe_record(out, {a}, [a, ad](const Tensor& g) {
        a->accumulate_grad(unbroadcast(nn::mul(g, mul_scalar(ad, 2.0f)), a->data.shape()));
    });
    return out;
}

// --- reductions ----------------------------------------------------------

Var sum_all(Var a) {
    auto out = make_var(nn::sum_all(a->data));
    maybe_record(out, {a}, [a](const Tensor& g) {
        a->accumulate_grad(Tensor::full(a->data.shape(), g.at({})));
    });
    return out;
}

Var mean_all(Var a) {
    auto out = make_var(nn::mean_all(a->data));
    const float n = static_cast<float>(a->data.numel());
    maybe_record(out, {a}, [a, n](const Tensor& g) {
        a->accumulate_grad(Tensor::full(a->data.shape(), g.at({}) / n));
    });
    return out;
}

Var softmax_rows_v(Var a) {
    auto out = make_var(softmax_rows(a->data));
    const Tensor od = out->data;
    maybe_record(out, {a}, [a, od](const Tensor& g) {
        // Jacobian-vector product: s * (g - sum(g*s, axis=1, keep))
        const int64_t n = od.dim(0), c = od.dim(1);
        Tensor ga(od.shape());
        for (int64_t i = 0; i < n; ++i) {
            float dot = 0.0f;
            for (int64_t j = 0; j < c; ++j) dot += g.at({i, j}) * od.at({i, j});
            for (int64_t j = 0; j < c; ++j)
                ga.at({i, j}) = od.at({i, j}) * (g.at({i, j}) - dot);
        }
        a->accumulate_grad(unbroadcast(ga, a->data.shape()));
    });
    return out;
}

// --- fused linear ---------------------------------------------------------

Var linear(Var x, Var w, Var b) {
    // out = x @ w + b, with b broadcast across rows
    auto out = make_var(nn::add(nn::matmul(x->data, w->data),
                                b->data.broadcast_to({x->data.dim(0), b->data.dim(0)})));
    const Tensor xd = x->data, wd = w->data;
    const int64_t n = x->data.dim(0);
    maybe_record(out, {x, w, b}, [x, w, b, xd, wd, n](const Tensor& g) {
        x->accumulate_grad(nn::matmul(g, wd.transpose2d()));
        w->accumulate_grad(nn::matmul(xd.transpose2d(), g));
        // db = sum of grad over batch rows -> shape [out]
        b->accumulate_grad(sum_rows(g).reshape(b->data.shape()));
    });
    return out;
}

// --- losses ---------------------------------------------------------------

Var mse_loss(Var pred, Var target) {
    Var diff = sub(pred, target);
    return mean_all(mul(diff, diff));
}

Var cross_entropy(Var logits, const std::vector<int>& labels) {
    const Tensor logp = log_softmax_rows(logits->data);
    const int64_t n = logits->data.dim(0);
    if (static_cast<int64_t>(labels.size()) != n)
        throw std::invalid_argument("cross_entropy: labels size mismatch");
    float loss = 0.0f;
    for (int64_t i = 0; i < n; ++i) loss -= logp.at({i, labels[static_cast<size_t>(i)]});
    loss /= static_cast<float>(n);
    auto out = make_var(Tensor::full({}, loss));
    const Tensor saved = logits->data; // for softmax in backward
    maybe_record(out, {logits}, [logits, saved, labels, n](const Tensor& g) {
        const float scale = g.at({}) / static_cast<float>(n);
        Tensor probs = softmax_rows(saved);
        for (int64_t i = 0; i < n; ++i)
            probs.at({i, labels[static_cast<size_t>(i)]}) -= 1.0f;
        for (auto& v : probs.data_) v *= scale;
        logits->accumulate_grad(probs);
    });
    return out;
}
} // namespace nn
