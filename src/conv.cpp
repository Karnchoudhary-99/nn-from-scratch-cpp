// Conv2d + MaxPool2d implementation: direct convolution with explicit
// backward passes, plus a MaxPool that routes gradients through argmax.
#include "nn/conv.hpp"

#include <cmath>
#include <stdexcept>

namespace nn {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride,
               int padding)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding) {
    // Kaiming (He) init for ReLU-ish activations.
    const float fan_in =
        static_cast<float>(in_channels * kernel_size * kernel_size);
    const float std = std::sqrt(2.0f / fan_in);
    weight = make_var(mul_scalar(Tensor::randn({out_channels, in_channels,
                                               kernel_size, kernel_size}),
                                std));
    bias = make_var(Tensor::zeros({out_channels}));
}

Var Conv2d::forward(Var x) {
    if (x->data.ndim() != 4 || x->data.dim(1) != in_channels)
        throw std::invalid_argument("Conv2d: expected [N, C_in, H, W]");
    return conv2d_v(x, weight, bias, stride, padding);
}

MaxPool2d::MaxPool2d(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride < 0 ? kernel_size : stride) {}

Var MaxPool2d::forward(Var x) {
    if (x->data.ndim() != 4)
        throw std::invalid_argument("MaxPool2d: expected [N, C, H, W]");
    return maxpool2d_v(x, kernel_size, stride);
}

} // namespace nn
