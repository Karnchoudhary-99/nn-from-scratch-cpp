// mnist_demo: train an MLP on MNIST (IDX files under data/, see README)
// and report test accuracy. data/ is gitignored; download once with:
//   ./scripts/download_mnist.sh   (or fetch from https://storage.googleapis.com/cvdf-datasets/mnist/)
#include "nn/autograd.hpp"
#include "nn/data.hpp"
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/serialization.hpp"
#include "nn/trainer.hpp"

#include <cstdio>

int main() {
    nn::set_seed(2026);

    nn::Dataset train, test;
    try {
        train = nn::load_mnist_idx("data/train-images-idx3-ubyte",
                                   "data/train-labels-idx1-ubyte");
        test = nn::load_mnist_idx("data/t10k-images-idx3-ubyte",
                                  "data/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::printf("MNIST data not found in data/ (%s)\n", e.what());
        std::printf("Download the IDX files first — see README.\n");
        return 2;
    }
    std::printf("train: %zu images, test: %zu images\n", train.size(),
                test.size());

    nn::Sequential model;
    model.add<nn::Linear>(784, 128);
    model.add<nn::LayerNorm>(128);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(128, 64);
    model.add<nn::LayerNorm>(64);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(64, 10);

    nn::DataLoader loader(train, 128, true);
    nn::Adam opt(model.parameters(), 3e-3f);
    nn::CosineAnnealingLR sched(opt, /*t_max=*/5);

    nn::TrainConfig cfg;
    cfg.epochs = 5;
    cfg.log_every = 1;
    nn::History h = nn::train_classifier(
        model, loader, opt,
        [](nn::Var logits, const std::vector<int>& y) {
            return nn::cross_entropy(logits, y);
        },
        cfg);
    sched.step();

    const float test_acc = nn::accuracy(model, test);
    std::printf("MNIST test accuracy: %.2f%%\n", test_acc * 100.0f);

    // Show a few predictions.
    model.eval();
    for (int i = 0; i < 8; ++i) {
        nn::Tensor x({1, 784});
        for (int64_t j = 0; j < 784; ++j) x.at({0, j}) = test.xs[i][j];
        nn::Var logits = model.forward(nn::constant(x));
        std::printf("digit %d -> predicted %lld\n", test.ys[i],
                    static_cast<long long>(nn::argmax_row(logits->data, 0)));
    }

    nn::save_model(model, "/tmp/mnist_mlp.nns1");
    std::printf("checkpoint saved to /tmp/mnist_mlp.nns1\n");
    return test_acc > 0.90f ? 0 : 1;
}
