// Optimizer implementations. Parameters are updated in place via
// their data tensors; gradients are read from the autograd engine.
#include "nn/optimizers.hpp"

#include <cmath>

namespace nn {

SGD::SGD(std::vector<Var> p, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(p)), lr(lr), momentum(momentum), weight_decay(weight_decay) {}

void SGD::step() {
    for (auto& p : params) {
        if (p->grad.numel() == 0) continue;
        Tensor g = p->grad;
        if (weight_decay != 0.0f) g = add(g, mul_scalar(p->data, weight_decay));
        if (momentum != 0.0f) {
            auto it = velocity_.find(p.get());
            if (it == velocity_.end())
                velocity_[p.get()] = g;
            else
                it->second = add(mul_scalar(it->second, momentum), g);
            g = velocity_[p.get()];
        }
        p->data = sub(p->data, mul_scalar(g, lr));
    }
}

Adam::Adam(std::vector<Var> p, float lr, float beta1, float beta2, float eps,
           float weight_decay)
    : Optimizer(std::move(p)), lr(lr), beta1(beta1), beta2(beta2), eps(eps),
      weight_decay(weight_decay) {}

void Adam::step() {
    ++t_;
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t_));
    for (auto& p : params) {
        if (p->grad.numel() == 0) continue;
        Tensor g = p->grad;
        if (weight_decay != 0.0f) g = add(g, mul_scalar(p->data, weight_decay));

        auto& m = m_[p.get()]; // default-constructs empty Tensor on first use
        auto& v = v_[p.get()];
        if (m.numel() == 0) {
            m = Tensor::zeros(p->data.shape());
            v = Tensor::zeros(p->data.shape());
        }
        m = add(mul_scalar(m, beta1), mul_scalar(g, 1.0f - beta1));
        v = add(mul_scalar(v, beta2), mul_scalar(pow2(g), 1.0f - beta2));

        Tensor mhat = mul_scalar(m, 1.0f / bc1);
        Tensor vhat = mul_scalar(v, 1.0f / bc2);
        Tensor denom = add_scalar(sqrt_t(vhat), eps);
        p->data = sub(p->data, mul_scalar(div(mhat, denom), lr));
    }
}

} // namespace nn
