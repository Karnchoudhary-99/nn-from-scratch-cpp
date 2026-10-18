// xor_demo: train a small MLP to solve XOR — the "hello world" of
// backpropagation. Expects ~100% accuracy in a few seconds.
#include "nn/data.hpp"
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/trainer.hpp"

#include <cstdio>

int main() {
    nn::set_seed(42);

    nn::Dataset train = nn::make_xor(400);
    nn::Dataset test = nn::make_xor(200);

    nn::Sequential model;
    model.add<nn::Linear>(2, 16);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(16, 16);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(16, 2);

    nn::Adam opt(model.parameters(), 0.01f);
    nn::DataLoader loader(train, 32);

    std::printf("Training MLP on XOR (%zu samples)...\n", train.size());
    nn::TrainConfig cfg;
    cfg.epochs = 200;
    cfg.log_every = 40;
    nn::train_classifier(model, loader, opt, nn::cross_entropy, cfg);

    const float acc = nn::accuracy(model, test);
    std::printf("test accuracy: %.4f\n", acc);
    return acc > 0.95f ? 0 : 1;
}
