// Layers implementation.
#include "nn/layers.hpp"

#include <cmath>
#include <random>

namespace nn {
namespace {
std::mt19937& dropout_rng() {
    static std::mt19937 gen(1234);
    return gen;
}
} // namespace

Linear::Linear(int in_features, int out_features) {
    // Kaiming/He init for ReLU-ish nets: w ~ N(0, sqrt(2/fan_in)).
    const float std = std::sqrt(2.0f / static_cast<float>(in_features));
    weight = make_var(Tensor::randn({in_features, out_features}, 0.0f, std));
    bias = make_var(Tensor::zeros({out_features}));
}

Var Linear::forward(Var x) { return linear(x, weight, bias); }

Dropout::Dropout(float prob) : p(prob) {}

Var Dropout::forward(Var x) {
    if (!training || p <= 0.0f) return x;
    if (p >= 1.0f) return mul_scalar(x, 0.0f);
    Tensor mask = Tensor::randu(x->data.shape());
    const float keep = 1.0f - p;
    for (auto& v : mask.data_) v = (v < keep ? 1.0f : 0.0f) / keep; // inverted dropout
    // Mask is a constant w.r.t. the graph (dropped units get zero gradient).
    return mul(x, constant(mask));
}

Var Sequential::forward(Var x) {
    for (auto& m : layers) x = m->forward(x);
    return x;
}

std::vector<Var> Sequential::parameters() {
    std::vector<Var> ps;
    for (auto& m : layers)
        for (auto& p : m->parameters()) ps.push_back(p);
    return ps;
}

void Sequential::train() {
    Module::train();
    for (auto& m : layers) m->train();
}

void Sequential::eval() {
    Module::eval();
    for (auto& m : layers) m->eval();
}

} // namespace nn
