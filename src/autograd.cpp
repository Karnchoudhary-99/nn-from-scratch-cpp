// Autograd engine implementation.
#include "nn/autograd.hpp"

#include <cmath>
#include <limits>
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
        const float* s = od.data_.data();
        const float* gg = g.data_.data();
        float* dst = ga.data_.data();
        for (int64_t i = 0; i < n; ++i) {
            float dot = 0.0f;
            for (int64_t j = 0; j < c; ++j) dot += gg[i * c + j] * s[i * c + j];
            for (int64_t j = 0; j < c; ++j)
                dst[i * c + j] = s[i * c + j] * (gg[i * c + j] - dot);
        }
        a->accumulate_grad(unbroadcast(ga, a->data.shape()));
    });
    return out;
}

// --- layer norm ------------------------------------------------------------

Var layernorm_v(Var x, Var gamma, Var beta, float eps) {
    const Tensor& xd = x->data;
    if (xd.ndim() != 2)
        throw std::invalid_argument("layernorm_v: only [N, C] supported");
    const int64_t n = xd.dim(0), c = xd.dim(1);
    if (gamma->data.numel() != c || beta->data.numel() != c)
        throw std::invalid_argument("layernorm_v: gamma/beta must be [C]");
    const float* src = xd.data_.data();
    const float* gd_ = gamma->data.data_.data();
    const float* bd = beta->data.data_.data();

    Tensor mu({n}), inv({n}), xh({n, c});
    Tensor out({n, c});
    float* od = out.data_.data();
    float* xhd = xh.data_.data();
    for (int64_t i = 0; i < n; ++i) {
        float m = 0.0f;
        for (int64_t j = 0; j < c; ++j) m += src[i * c + j];
        m /= static_cast<float>(c);
        float v = 0.0f;
        for (int64_t j = 0; j < c; ++j) {
            float d = src[i * c + j] - m;
            v += d * d;
        }
        v /= static_cast<float>(c);
        const float iv = 1.0f / std::sqrt(v + eps);
        mu[i] = m;
        inv[i] = iv;
        for (int64_t j = 0; j < c; ++j) {
            const float xhn = (src[i * c + j] - m) * iv;
            xhd[i * c + j] = xhn;
            od[i * c + j] = xhn * gd_[j] + bd[j];
        }
    }
    auto outv = make_var(std::move(out));
    maybe_record(outv, {x, gamma, beta},
                 [x, gamma, beta, mu, inv, xh, n, c](const Tensor& g) {
                     const float* gd = g.data_.data();
                     const float* xhd = xh.data_.data();
                     const float* gm = gamma->data.data_.data();
                     const float* iv = inv.data_.data();
                     Tensor dx({n, c}), dgm({c}), dbt({c});
                     float* dxd = dx.data_.data();
                     float* dgd = dgm.data_.data();
                     float* dbd = dbt.data_.data();
                     // dL/dx per row: (1/s)(a - mean(a) - xh*mean(a*xh)), a = g*gamma
                     for (int64_t i = 0; i < n; ++i) {
                         float mean_a = 0.0f, mean_ax = 0.0f;
                         for (int64_t j = 0; j < c; ++j) {
                             const float a = gd[i * c + j] * gm[j];
                             mean_a += a;
                             mean_ax += a * xhd[i * c + j];
                         }
                         mean_a /= static_cast<float>(c);
                         mean_ax /= static_cast<float>(c);
                         for (int64_t j = 0; j < c; ++j)
                             dxd[i * c + j] = (gd[i * c + j] * gm[j] - mean_a -
                                               xhd[i * c + j] * mean_ax) *
                                              iv[i];
                     }
                     // dL/dgamma = sum_n g*xh, dL/dbeta = sum_n g
                     for (int64_t j = 0; j < c; ++j) {
                         float s1 = 0.0f, s2 = 0.0f;
                         for (int64_t i = 0; i < n; ++i) {
                             s1 += gd[i * c + j] * xhd[i * c + j];
                             s2 += gd[i * c + j];
                         }
                         dgd[j] = s1;
                         dbd[j] = s2;
                     }
                     x->accumulate_grad(dx);
                     gamma->accumulate_grad(dgm);
                     beta->accumulate_grad(dbt);
                 });
    return outv;
}

// --- reshape / conv2d / maxpool --------------------------------------------

Var reshape_v(Var a, std::vector<int64_t> shape) {
    if (numel_of(shape) != a->data.numel())
        throw std::invalid_argument("reshape_v: element count mismatch");
    const std::vector<int64_t> in_shape = a->data.shape();
    auto outv = make_var(a->data.reshape(shape));
    maybe_record(outv, {a}, [a, in_shape](const Tensor& g) {
        a->accumulate_grad(g.reshape(in_shape));
    });
    return outv;
}

namespace {

int64_t conv_out_size(int64_t in, int64_t k, int64_t stride, int64_t pad) {
    return (in + 2 * pad - k) / stride + 1;
}

} // namespace

Var conv2d_v(Var x, Var w, Var b, int stride, int padding) {
    const Tensor& xd = x->data;
    const Tensor& wd = w->data;
    const int64_t n = xd.dim(0), ic = xd.dim(1), h = xd.dim(2), ww = xd.dim(3);
    const int64_t oc = wd.dim(0), k = wd.dim(2);
    const int64_t oh = conv_out_size(h, k, stride, padding);
    const int64_t ow = conv_out_size(ww, k, stride, padding);
    if (oh <= 0 || ow <= 0)
        throw std::invalid_argument("conv2d_v: kernel larger than input");

    const float* xp = xd.data_.data();
    const float* wpp = wd.data_.data();
    const float* bp = b->data.data_.data();

    // Zero-padded input: [N, IC, h+2p, w+2p].
    const int64_t hp = h + 2 * padding, wpd = ww + 2 * padding;
    Tensor xpad({n, ic, hp, wpd});
    float* xpp = xpad.data_.data();
    for (int64_t bn = 0; bn < n; ++bn)
        for (int64_t c = 0; c < ic; ++c)
            for (int64_t i = 0; i < h; ++i)
                for (int64_t j = 0; j < ww; ++j)
                    xpp[((bn * ic + c) * hp + (i + padding)) * wpd +
                        (j + padding)] = xp[((bn * ic + c) * h + i) * ww + j];

    Tensor out({n, oc, oh, ow});
    float* op = out.data_.data();
    for (int64_t bn = 0; bn < n; ++bn)
        for (int64_t o = 0; o < oc; ++o)
            for (int64_t i = 0; i < oh; ++i)
                for (int64_t j = 0; j < ow; ++j) {
                    float acc = bp[o];
                    for (int64_t c = 0; c < ic; ++c) {
                        const int64_t xbase =
                            ((bn * ic + c) * hp + i * stride) * wpd + j * stride;
                        const int64_t wbase = ((o * ic + c) * k) * k;
                        for (int64_t a = 0; a < k; ++a)
                            for (int64_t q = 0; q < k; ++q)
                                acc += xpp[xbase + a * wpd + q] *
                                       wpp[wbase + a * k + q];
                    }
                    op[((bn * oc + o) * oh + i) * ow + j] = acc;
                }

    auto outv = make_var(std::move(out));
    maybe_record(outv, {x, w, b},
                 [x, w, b, xpad, n, ic, oc, h, ww, hp, wpd, k, stride, padding, oh,
                  ow](const Tensor& g) {
                     const float* gd = g.data_.data();
                     const float* xpp = xpad.data_.data();
                     const float* wpp = w->data.data_.data();

                     Tensor dxpad({n, ic, hp, wpd});
                     float* dxp = dxpad.data_.data();
                     Tensor dw({oc, ic, k, k});
                     float* dwp = dw.data_.data();
                     Tensor db({oc});
                     float* dbp = db.data_.data();

                     for (int64_t bn = 0; bn < n; ++bn)
                         for (int64_t o = 0; o < oc; ++o)
                             for (int64_t i = 0; i < oh; ++i)
                                 for (int64_t j = 0; j < ow; ++j) {
                                     const float gv =
                                         gd[((bn * oc + o) * oh + i) * ow + j];
                                     dbp[o] += gv;
                                     for (int64_t c = 0; c < ic; ++c) {
                                         const int64_t xbase =
                                             ((bn * ic + c) * hp + i * stride) *
                                                 wpd +
                                             j * stride;
                                         const int64_t wbase =
                                             ((o * ic + c) * k) * k;
                                         for (int64_t a = 0; a < k; ++a)
                                             for (int64_t q = 0; q < k; ++q) {
                                                 dwp[wbase + a * k + q] +=
                                                     gv * xpp[xbase + a * wpd +
                                                              q];
                                                 dxp[xbase + a * wpd + q] +=
                                                     gv * wpp[wbase + a * k + q];
                                             }
                                     }
                                 }
                     // Strip the padding from dx.
                     Tensor dx({n, ic, h, ww});
                     float* dxd = dx.data_.data();
                     for (int64_t bn = 0; bn < n; ++bn)
                         for (int64_t c = 0; c < ic; ++c)
                             for (int64_t i = 0; i < h; ++i)
                                 for (int64_t j = 0; j < ww; ++j)
                                     dxd[((bn * ic + c) * h + i) * ww + j] =
                                         dxp[((bn * ic + c) * hp +
                                              (i + padding)) *
                                                 wpd +
                                             (j + padding)];
                     x->accumulate_grad(dx);
                     w->accumulate_grad(dw);
                     b->accumulate_grad(db);
                 });
    return outv;
}

Var maxpool2d_v(Var x, int kernel, int stride) {
    const Tensor& xd = x->data;
    const int64_t n = xd.dim(0), c = xd.dim(1), h = xd.dim(2), w = xd.dim(3);
    const int64_t oh = (h - kernel) / stride + 1;
    const int64_t ow = (w - kernel) / stride + 1;
    if (oh <= 0 || ow <= 0)
        throw std::invalid_argument("maxpool2d_v: kernel larger than input");

    const float* xp = xd.data_.data();
    Tensor out({n, c, oh, ow});
    Tensor argmax({n, c, oh, ow}); // flat index of max within the window
    float* op = out.data_.data();
    float* amp = argmax.data_.data();

    for (int64_t bn = 0; bn < n; ++bn)
        for (int64_t ch = 0; ch < c; ++ch)
            for (int64_t i = 0; i < oh; ++i)
                for (int64_t j = 0; j < ow; ++j) {
                    float best = -std::numeric_limits<float>::infinity();
                    int64_t bi = 0;
                    for (int64_t a = 0; a < kernel; ++a)
                        for (int64_t q = 0; q < kernel; ++q) {
                            const float v =
                                xp[((bn * c + ch) * h + i * stride + a) * w +
                                   j * stride + q];
                            if (v > best) {
                                best = v;
                                bi = a * kernel + q;
                            }
                        }
                    op[((bn * c + ch) * oh + i) * ow + j] = best;
                    amp[((bn * c + ch) * oh + i) * ow + j] =
                        static_cast<float>(bi);
                }

    auto outv = make_var(std::move(out));
    maybe_record(outv, {x},
                 [x, argmax, n, c, h, w, kernel, stride, oh,
                  ow](const Tensor& g) {
                     Tensor dx({n, c, h, w});
                     const float* gd = g.data_.data();
                     const float* amp = argmax.data_.data();
                     float* dxd = dx.data_.data();
                     for (int64_t bn = 0; bn < n; ++bn)
                         for (int64_t ch = 0; ch < c; ++ch)
                             for (int64_t i = 0; i < oh; ++i)
                                 for (int64_t j = 0; j < ow; ++j) {
                                     const int64_t bi = static_cast<int64_t>(
                                         amp[((bn * c + ch) * oh + i) * ow +
                                             j]);
                                     const int64_t a = bi / kernel;
                                     const int64_t q = bi % kernel;
                                     dxd[((bn * c + ch) * h + i * stride + a) *
                                             w +
                                         j * stride + q] +=
                                         gd[((bn * c + ch) * oh + i) * ow + j];
                                 }
                     x->accumulate_grad(dx);
                 });
    return outv;
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
        float* pd = probs.data_.data();
        const int64_t c = probs.dim(1);
        for (int64_t i = 0; i < n; ++i) pd[i * c + labels[static_cast<size_t>(i)]] -= 1.0f;
        for (auto& v : probs.data_) v *= scale;
        logits->accumulate_grad(probs);
    });
    return out;
}
} // namespace nn
