// Schedulers: learning-rate schedules wrapping an Optimizer.
// Call step() once per epoch (or per batch, depending on the schedule).
#pragma once

#include "nn/optimizers.hpp"

namespace nn {

// Decays the learning rate by `gamma` every `step_size` steps:
// lr = lr0 * gamma^(floor(steps / step_size)).
struct StepLR {
    Optimizer& opt;
    float lr0;
    int step_size;
    float gamma;
    int steps = 0;

    StepLR(Optimizer& o, int step_size, float gamma = 0.1f);
    void step();
};

// Cosine annealing from lr0 down to lr_min over T_max steps:
// lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + cos(pi * t / T_max)).
struct CosineAnnealingLR {
    Optimizer& opt;
    float lr0;
    float lr_min;
    int t_max;
    int steps = 0;

    CosineAnnealingLR(Optimizer& o, int t_max, float lr_min = 0.0f);
    void step();
};

// Exponential decay: lr = lr0 * gamma^steps.
struct ExponentialLR {
    Optimizer& opt;
    float lr0;
    float gamma;
    int steps = 0;

    ExponentialLR(Optimizer& o, float gamma);
    void step();
};

} // namespace nn
