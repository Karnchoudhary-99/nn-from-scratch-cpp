// spiral_demo: a harder non-linear problem — two interleaving spirals.
// Needs a wider network and more epochs than XOR.
#include "nn/data.hpp"
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/trainer.hpp"

#include <cstdio>

int main() {
    nn::set_seed(7);

    nn::Dataset train = nn::make_spirals(600);
    nn::Dataset test = nn::make_spirals(200);

    nn::Sequential model;
    model.add<nn::Linear>(2, 64);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(64, 64);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(64, 2);

    nn::Adam opt(model.parameters(), 0.005f);
    nn::DataLoader loader(train, 64);

    std::printf("Training MLP on two spirals (%zu samples)...\n", train.size());
    nn::TrainConfig cfg;
    cfg.epochs = 400;
    cfg.log_every = 80;
    nn::train_classifier(model, loader, opt, nn::cross_entropy, cfg);

    const float acc = nn::accuracy(model, test);
    std::printf("test accuracy: %.4f\n", acc);
    return acc > 0.90f ? 0 : 1;
}
