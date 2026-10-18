// Learning-rate scheduler implementations.
#include "nn/schedulers.hpp"

#include <cmath>

namespace nn {

StepLR::StepLR(Optimizer& o, int step_size, float gamma)
    : opt(o), lr0(o.get_lr()), step_size(step_size), gamma(gamma) {}

void StepLR::step() {
    ++steps;
    opt.set_lr(lr0 * std::pow(gamma, static_cast<float>(steps / step_size)));
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& o, int t_max, float lr_min)
    : opt(o), lr0(o.get_lr()), lr_min(lr_min), t_max(t_max) {}

void CosineAnnealingLR::step() {
    ++steps;
    int t = steps < t_max ? steps : t_max;
    float f = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * t / t_max));
    opt.set_lr(lr_min + (lr0 - lr_min) * f);
}

ExponentialLR::ExponentialLR(Optimizer& o, float gamma)
    : opt(o), lr0(o.get_lr()), gamma(gamma) {}

void ExponentialLR::step() {
    ++steps;
    opt.set_lr(lr0 * std::pow(gamma, static_cast<float>(steps)));
}

} // namespace nn
