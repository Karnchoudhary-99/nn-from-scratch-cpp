// mnist_cnn: train the MnistCNN convolutional classifier on MNIST and report
// test accuracy. data/ is gitignored; download once with ./scripts/download_mnist.sh
//
// Quick smoke mode for CI:  MNIST_CNN_QUICK=1 ./build/mnist_cnn
//   (1 epoch on a 10k subset, accuracy gate 80%)
// Full run (default): 3 epochs on all 60k images, accuracy gate 97%.
// Overrides: MNIST_CNN_EPOCHS, MNIST_CNN_SUBSET.
#include "nn/autograd.hpp"
#include "nn/data.hpp"
#include "nn/optimizers.hpp"
#include "nn/serialization.hpp"
#include "nn/trainer.hpp"
#include "mnist_cnn_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}

} // namespace

int main() {
    nn::set_seed(2026);

    nn::Dataset train_full, test;
    try {
        train_full = nn::load_mnist_idx("data/train-images-idx3-ubyte",
                                        "data/train-labels-idx1-ubyte");
        test = nn::load_mnist_idx("data/t10k-images-idx3-ubyte",
                                  "data/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::printf("MNIST data not found in data/ (%s)\n", e.what());
        std::printf("Download the IDX files first — see README.\n");
        return 2;
    }

    const bool quick = env_int("MNIST_CNN_QUICK", 0) == 1;
    const int epochs = quick ? 1 : env_int("MNIST_CNN_EPOCHS", 3);
    const size_t subset =
        quick ? 10000u : static_cast<size_t>(env_int("MNIST_CNN_SUBSET", 60000));
    const float gate = quick ? 0.80f : 0.97f;

    nn::Dataset train;
    train.num_classes = 10;
    for (size_t i = 0; i < train_full.size() && i < subset; ++i) {
        train.xs.push_back(train_full.xs[i]);
        train.ys.push_back(train_full.ys[i]);
    }
    std::printf("train: %zu images (%s mode), test: %zu images\n", train.size(),
                quick ? "quick" : "full", test.size());

    nn::MnistCNN model;
    nn::DataLoader loader(train, 64, true);
    nn::Adam opt(model.parameters(), 1e-3f);

    nn::TrainConfig cfg;
    cfg.epochs = epochs;
    cfg.log_every = 1;
    nn::History h = nn::train_classifier(
        model, loader, opt,
        [](nn::Var logits, const std::vector<int>& y) {
            return nn::cross_entropy(logits, y);
        },
        cfg);

    const float test_acc = nn::accuracy(model, test);
    std::printf("MNIST CNN test accuracy: %.2f%%\n", test_acc * 100.0f);

    // Show a few predictions.
    model.eval();
    for (int i = 0; i < 8; ++i) {
        nn::Tensor x({1, 784});
        for (int64_t j = 0; j < 784; ++j) x.at({0, j}) = test.xs[i][j];
        nn::Var logits = model.forward(nn::constant(x));
        std::printf("digit %d -> predicted %lld\n", test.ys[i],
                    static_cast<long long>(nn::argmax_row(logits->data, 0)));
    }

    nn::save_model(model, "/tmp/mnist_cnn.nns1");
    std::printf("checkpoint saved to /tmp/mnist_cnn.nns1\n");
    return test_acc > gate ? 0 : 1;
}
