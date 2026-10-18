// Trainer implementation.
#include "nn/trainer.hpp"

#include <cstdio>

namespace nn {

History train_classifier(Module& model, DataLoader& loader, Optimizer& opt,
                         LossFn loss_fn, const TrainConfig& cfg) {
    History h;
    model.train();
    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        float total_loss = 0.0f;
        int total = 0, correct = 0;
        for (auto& batch : loader.batches()) {
            opt.zero_grad();
            Var x = make_var(batch.X);
            Var logits = model.forward(x);
            Var loss = loss_fn(logits, batch.y);
            loss->backward();
            opt.step();
            total_loss += loss->data.at({}) * static_cast<float>(batch.y.size());
            // batch accuracy from raw logits (no grad needed)
            for (size_t i = 0; i < batch.y.size(); ++i) {
                if (argmax_row(logits->data, static_cast<int64_t>(i)) == batch.y[i])
                    ++correct;
                ++total;
            }
        }
        const float epoch_loss = total_loss / static_cast<float>(total);
        const float epoch_acc = static_cast<float>(correct) / static_cast<float>(total);
        h.loss.push_back(epoch_loss);
        h.acc.push_back(epoch_acc);
        if (cfg.verbose && (epoch % cfg.log_every == 0 || epoch == 1 || epoch == cfg.epochs))
            std::printf("epoch %4d  loss %.4f  acc %.4f\n", epoch, epoch_loss, epoch_acc);
    }
    return h;
}

float accuracy(Module& model, const Dataset& ds) {
    model.eval();
    int correct = 0;
    for (size_t i = 0; i < ds.size(); ++i) {
        Var x = constant(ds.xs[i].reshape({1, ds.xs[i].dim(0)}));
        Var logits = model.forward(x);
        if (argmax_row(logits->data, 0) == ds.ys[i]) ++correct;
    }
    model.train();
    return static_cast<float>(correct) / static_cast<float>(ds.size());
}

} // namespace nn
