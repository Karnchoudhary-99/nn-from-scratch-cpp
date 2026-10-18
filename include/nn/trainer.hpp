// Trainer: mini-batch training loop with loss/accuracy tracking.
#pragma once

#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/data.hpp"

#include <functional>
#include <vector>

namespace nn {

using LossFn = std::function<Var(Var logits, const std::vector<int>& labels)>;

struct TrainConfig {
    int epochs = 100;
    int log_every = 10;
    bool verbose = true;
};

struct History {
    std::vector<float> loss; // per-epoch mean loss
    std::vector<float> acc;  // per-epoch accuracy
};

History train_classifier(Module& model, DataLoader& loader, Optimizer& opt,
                         LossFn loss_fn, const TrainConfig& cfg);

// Accuracy of `model` over a full dataset (model switched to eval mode).
float accuracy(Module& model, const Dataset& ds);

} // namespace nn
