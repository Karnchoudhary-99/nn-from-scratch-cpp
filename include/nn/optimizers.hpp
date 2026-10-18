// Optimizers: SGD (+momentum, weight decay) and Adam.
#pragma once

#include "nn/autograd.hpp"

#include <unordered_map>
#include <vector>

namespace nn {

struct Optimizer {
    std::vector<Var> params;
    explicit Optimizer(std::vector<Var> p) : params(std::move(p)) {}
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void zero_grad() {
        for (auto& p : params) p->zero_grad();
    }
};

struct SGD : Optimizer {
    float lr;
    float momentum;
    float weight_decay;

    SGD(std::vector<Var> p, float lr = 0.01f, float momentum = 0.0f,
        float weight_decay = 0.0f);
    void step() override;

private:
    std::unordered_map<Variable*, Tensor> velocity_;
};

struct Adam : Optimizer {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    Adam(std::vector<Var> p, float lr = 1e-3f, float beta1 = 0.9f,
         float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;

private:
    std::unordered_map<Variable*, Tensor> m_;
    std::unordered_map<Variable*, Tensor> v_;
    int64_t t_ = 0;
};

} // namespace nn
