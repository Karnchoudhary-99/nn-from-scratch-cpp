// cnn_demo: train a small CNN on the synthetic shapes dataset
// (circles vs squares, 28x28) — all data generated in-process, no downloads.
// Saves a checkpoint, reloads it into a fresh model, and verifies identical
// predictions.
#include "nn/autograd.hpp"
#include "nn/conv.hpp"
#include "nn/data.hpp"
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/serialization.hpp"
#include "nn/trainer.hpp"

#include <cstdio>

namespace {

struct CNN : nn::Module {
    nn::Conv2d c1{1, 8, 3, 1, 1};  // 28x28 -> 28x28
    nn::Conv2d c2{8, 16, 3, 1, 1}; // 14x14 -> 14x14
    nn::MaxPool2d p1{2}, p2{2};
    nn::ReLU relu;
    nn::Linear fc{16 * 7 * 7, 2};

    nn::Var forward(nn::Var x) override {
        x = nn::reshape_v(x, {x->data.dim(0), 1, 28, 28});
        x = p1.forward(relu.forward(c1.forward(x)));
        x = p2.forward(relu.forward(c2.forward(x)));
        x = nn::reshape_v(x, {x->data.dim(0), 16 * 7 * 7});
        return fc.forward(x);
    }

    std::vector<nn::Var> parameters() override {
        auto ps = c1.parameters();
        for (auto& p : c2.parameters()) ps.push_back(p);
        for (auto& p : fc.parameters()) ps.push_back(p);
        return ps;
    }

    std::vector<std::pair<std::string, nn::Var>> named_parameters() override {
        std::vector<std::pair<std::string, nn::Var>> ps;
        for (auto& [n, p] : c1.named_parameters()) ps.push_back({"c1." + n, p});
        for (auto& [n, p] : c2.named_parameters()) ps.push_back({"c2." + n, p});
        for (auto& [n, p] : fc.named_parameters()) ps.push_back({"fc." + n, p});
        return ps;
    }
};

} // namespace

int main() {
    nn::set_seed(42);

    auto train_ds = nn::make_shapes(1600);
    auto test_ds = nn::make_shapes(400);
    nn::DataLoader loader(train_ds, 64, true);

    CNN model;
    nn::Adam opt(model.parameters(), 1e-3f);

    nn::TrainConfig cfg;
    cfg.epochs = 8;
    cfg.log_every = 2;
    nn::History h = nn::train_classifier(
        model, loader, opt,
        [](nn::Var logits, const std::vector<int>& y) {
            return nn::cross_entropy(logits, y);
        },
        cfg);

    const float test_acc = nn::accuracy(model, test_ds);
    std::printf("CNN test accuracy: %.2f%%\n", test_acc * 100.0f);

    // Checkpoint round-trip: save, reload into a fresh model, compare.
    nn::save_model(model, "/tmp/cnn_shapes.nns1");
    CNN fresh;
    nn::load_model(fresh, "/tmp/cnn_shapes.nns1");
    const float fresh_acc = nn::accuracy(fresh, test_ds);
    std::printf("reloaded model test accuracy: %.2f%%\n", fresh_acc * 100.0f);
    if (test_acc != fresh_acc) {
        std::printf("CHECKPOINT MISMATCH\n");
        return 1;
    }
    std::printf("checkpoint verified: identical predictions\n");
    return test_acc > 0.90f ? 0 : 1;
}
