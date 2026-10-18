// Autograd: reverse-mode automatic differentiation over a dynamic
// computation graph (micrograd-style). Every differentiable op records a
// Node; Variable::backward() topologically sorts the graph and applies the
// chain rule.
#pragma once

#include "nn/tensor.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace nn {

struct Variable;

// A recorded operation: its inputs and how to propagate a gradient.
struct Node {
    std::vector<std::shared_ptr<Variable>> inputs;
    // Given dL/d(output), accumulate dL/d(input) into each input's grad.
    std::function<void(const Tensor& grad_out)> backward_fn;
};

struct Variable {
    Tensor data;
    Tensor grad; // same shape as data; empty until backward() runs
    std::shared_ptr<Node> creator; // null for leaves / constants
    bool requires_grad = true;

    explicit Variable(Tensor d, bool req = true)
        : data(std::move(d)), requires_grad(req) {}

    // Backpropagate from this (scalar) variable through the whole graph.
    void backward();
    void zero_grad() { grad = Tensor(); }

    // dL/dself += g (handles the first accumulation specially).
    void accumulate_grad(const Tensor& g);
};

using Var = std::shared_ptr<Variable>;

Var make_var(Tensor t, bool requires_grad = true);
Var constant(Tensor t); // leaf that never accumulates a gradient

// --- Differentiable ops -------------------------------------------------
Var add(Var a, Var b);
Var sub(Var a, Var b);
Var mul(Var a, Var b);
Var div(Var a, Var b);
Var neg(Var a);
Var add_scalar(Var a, float s);
Var mul_scalar(Var a, float s);

Var matmul(Var a, Var b); // 2-D

Var relu(Var a);
Var sigmoid(Var a);
Var tanh_act(Var a);
Var exp_v(Var a);
Var log_v(Var a);
Var pow2_v(Var a);

Var sum_all(Var a);
Var mean_all(Var a);
Var softmax_rows_v(Var a);

// Fused linear layer op: out = x @ w + b (b broadcast over rows).
Var linear(Var x, Var w, Var b);

// --- Losses (scalar outputs) --------------------------------------------
Var mse_loss(Var pred, Var target); // mean((pred - target)^2), broadcastable
Var cross_entropy(Var logits, const std::vector<int>& labels); // [N, C] logits

} // namespace nn
