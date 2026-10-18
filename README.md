# Neural Network from Scratch (C++)

Build a **complete neural network framework in modern C++** — no PyTorch,
no TensorFlow, no Eigen. Every line of the math is ours: tensors, an
autograd engine, layers, losses, and optimizers, trained on MNIST until it
actually learns.

## Why this exists

Frameworks hide the machinery. Rebuilding it by hand — reverse-mode
automatic differentiation, backprop through a computation graph, SGD with
momentum — is the fastest way to *actually* understand deep learning, and a
C++ implementation shows systems-level ownership of the math.

## Roadmap

- [x] `Tensor`: N-dimensional array, row-major storage, broadcasting, matmul
- [x] Autograd: computation graph, reverse-mode differentiation
- [x] Layers: Linear, ReLU/Sigmoid/Tanh/Softmax, Dropout, Sequential
- [x] Losses: MSE, cross-entropy (with log-softmax fusion)
- [x] Optimizers: SGD (+momentum, weight decay), Adam
- [x] Data: synthetic datasets (XOR, moons, spirals, blobs) + shuffling DataLoader
- [x] Training loop: mini-batch, shuffling, accuracy metrics
- [x] Demos: XOR (100% test acc), spiral classification (98.5% test acc)
- [x] Tests: 17 tests — tensor unit tests, numeric gradient checks for every op, training convergence
- [ ] Data: MNIST loader (IDX format, no dependencies)
- [ ] Demos: MNIST digit classifier
- [ ] Benchmarks: forward/backward throughput vs. a Python baseline

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build          # 17 tests, all green
./build/xor_demo                # MLP learns XOR → 100% test accuracy
./build/spiral_demo             # MLP learns two spirals → ~98% test accuracy
```

## Quick start

```cpp
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/trainer.hpp"
#include "nn/data.hpp"

nn::Sequential model;
model.add<nn::Linear>(2, 16);
model.add<nn::ReLU>();
model.add<nn::Linear>(16, 2);

nn::Adam opt(model.parameters(), 0.01f);
nn::Dataset train = nn::make_xor(400);
nn::DataLoader loader(train, 32);
nn::TrainConfig cfg{.epochs = 200};
nn::train_classifier(model, loader, opt, nn::cross_entropy, cfg);
```

## Project layout

```
include/nn/     public headers (tensor, autograd, layers, optimizers)
src/            implementations
tests/          gradient checks + unit tests (CTest)
examples/       xor_demo, spiral_demo, mnist_demo
data/           MNIST IDX files (downloaded, gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
