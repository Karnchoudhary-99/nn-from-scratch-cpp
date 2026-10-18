// Convolution and pooling: Conv2d + MaxPool2d over [N, C, H, W] tensors.
// Direct (im2col-free) implementation, fully differentiable.
#pragma once

#include "nn/autograd.hpp"
#include "nn/layers.hpp"

namespace nn {

// Conv2d: y[n, oc, oh, ow] = bias[oc] + sum_{ic,i,j} x[n, ic, oh*s-pad+i, ow*s-pad+j] * w[oc, ic, i, j]
// weight: [out_channels, in_channels, kernel, kernel] (Kaiming init), bias: [out_channels].
struct Conv2d : Module {
    int in_channels, out_channels, kernel_size, stride, padding;
    Var weight;
    Var bias;

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
           int padding = 0);
    Var forward(Var x) override;
    std::vector<Var> parameters() override { return {weight, bias}; }
    std::vector<std::pair<std::string, Var>> named_parameters() override {
        return {{"weight", weight}, {"bias", bias}};
    }
};

// MaxPool2d: max over each kernel window (non-overlapping by default).
struct MaxPool2d : Module {
    int kernel_size, stride;

    explicit MaxPool2d(int kernel_size, int stride = -1);
    Var forward(Var x) override; // [N, C, H, W] -> [N, C, H/stride, W/stride]
};

} // namespace nn
