// MnistCNN: a small convolutional digit classifier shared by the MNIST CNN
// training demo (examples/mnist_cnn.cpp) and the visualisation example
// (examples/visualize.cpp).
//
//   conv1: 1 -> 8 channels, 5x5, pad 2   (28x28 -> 28x28)
//   pool:  2x2                            (28x28 -> 14x14)
//   conv2: 8 -> 16 channels, 5x5, pad 2  (14x14 -> 14x14)
//   pool:  2x2                            (14x14 -> 7x7)
//   fc1: 16*7*7 -> 64, fc2: 64 -> 10
#pragma once

#include "nn/conv.hpp"
#include "nn/layers.hpp"

namespace nn {

struct MnistCNN : Module {
    Conv2d c1{1, 8, 5, 1, 2};
    MaxPool2d p1{2};
    Conv2d c2{8, 16, 5, 1, 2};
    MaxPool2d p2{2};
    ReLU relu;
    Linear fc1{16 * 7 * 7, 64};
    Linear fc2{64, 10};

    Var forward(Var x) override {
        x = nn::reshape_v(x, {x->data.dim(0), 1, 28, 28});
        x = relu.forward(c1.forward(x));
        x = p1.forward(x);
        x = relu.forward(c2.forward(x));
        x = p2.forward(x);
        x = reshape_v(x, {x->data.dim(0), 16 * 7 * 7});
        x = relu.forward(fc1.forward(x));
        return fc2.forward(x);
    }

    std::vector<Var> parameters() override {
        auto ps = c1.parameters();
        for (auto& p : c2.parameters()) ps.push_back(p);
        for (auto& p : fc1.parameters()) ps.push_back(p);
        for (auto& p : fc2.parameters()) ps.push_back(p);
        return ps;
    }

    std::vector<std::pair<std::string, Var>> named_parameters() override {
        std::vector<std::pair<std::string, Var>> ps;
        for (auto& [n, p] : c1.named_parameters()) ps.push_back({"c1." + n, p});
        for (auto& [n, p] : c2.named_parameters()) ps.push_back({"c2." + n, p});
        for (auto& [n, p] : fc1.named_parameters()) ps.push_back({"fc1." + n, p});
        for (auto& [n, p] : fc2.named_parameters()) ps.push_back({"fc2." + n, p});
        return ps;
    }
};

} // namespace nn
