// Layers: Module base, Linear, activations, Sequential, Dropout.
#pragma once

#include "nn/autograd.hpp"

#include <memory>
#include <vector>

namespace nn {

struct Module {
    bool training = true;
    virtual ~Module() = default;
    virtual Var forward(Var x) = 0;
    virtual std::vector<Var> parameters() { return {}; }

    void zero_grad() {
        for (auto& p : parameters()) p->zero_grad();
    }
    virtual void train() { training = true; }
    virtual void eval() { training = false; }
};

// Fully-connected layer: y = x @ W + b. W ~ Kaiming init.
struct Linear : Module {
    Var weight; // [in_features, out_features]
    Var bias;   // [out_features]

    Linear(int in_features, int out_features);
    Var forward(Var x) override;
    std::vector<Var> parameters() override { return {weight, bias}; }
};

struct ReLU : Module {
    Var forward(Var x) override { return relu(x); }
};
struct Sigmoid : Module {
    Var forward(Var x) override { return sigmoid(x); }
};
struct Tanh : Module {
    Var forward(Var x) override { return tanh_act(x); }
};
struct Softmax : Module {
    Var forward(Var x) override { return softmax_rows_v(x); }
};

// Randomly zeroes activations with probability p while training.
struct Dropout : Module {
    float p;
    explicit Dropout(float prob = 0.5f);
    Var forward(Var x) override;
};

// Chains modules: out = m_n(...m_2(m_1(x))).
struct Sequential : Module {
    std::vector<std::shared_ptr<Module>> layers;

    template <typename M, typename... Args>
    void add(Args&&... args) {
        layers.push_back(std::make_shared<M>(std::forward<Args>(args)...));
    }

    Var forward(Var x) override;
    std::vector<Var> parameters() override;
    void train() override;
    void eval() override;
};

} // namespace nn
