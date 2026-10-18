# Neural Network from Scratch (C++)

Build a **complete neural network framework in modern C++** — no PyTorch,
no TensorFlow, no Eigen. Every line of the math is ours: tensors, an
autograd engine, layers, losses, optimizers, convolution, normalisation,
checkpoints, and learning-rate schedules. It trains real models: an MLP
learns spirals at 98.5% and a small CNN hits 99%+ on a 28x28 shapes dataset.

## Why this exists

Frameworks hide the machinery. Rebuilding it by hand — reverse-mode
automatic differentiation, backprop through a computation graph, SGD with
momentum, backprop through a convolution — is the fastest way to *actually*
understand deep learning, and a C++ implementation shows systems-level
ownership of the math.

## Roadmap

- [x] `Tensor`: N-dimensional array, row-major storage, broadcasting, matmul
- [x] Autograd: computation graph, reverse-mode differentiation
- [x] Layers: Linear, ReLU/Sigmoid/Tanh/Softmax, Dropout, LayerNorm, Sequential
- [x] Conv: Conv2d (direct, Kaiming init), MaxPool2d, reshape — all gradchecked
- [x] Losses: MSE, cross-entropy (with log-softmax fusion)
- [x] Optimizers: SGD (+momentum, weight decay), Adam
- [x] Schedulers: StepLR, CosineAnnealingLR, ExponentialLR
- [x] Serialization: NNS1 binary checkpoints (save/load with shape checks)
- [x] Data: synthetic datasets (XOR, moons, spirals, blobs, 28x28 shapes) + shuffling DataLoader
- [x] Training loop: mini-batch, shuffling, accuracy metrics
- [x] Demos: XOR (100% test acc), spirals (98.5%), CNN on shapes (99.25% test acc)
- [x] Tests: 33 tests — tensor unit tests, numeric gradient checks for every op, training convergence
- [x] Benchmarks: forward/backward throughput per op (`examples/benchmark.cpp`)
- [x] Data: MNIST IDX loader (big-endian, no dependencies)
- [x] Demos: MNIST MLP classifier — 97.3% test accuracy in 61s (5 epochs)
- [ ] Demos: CNN digit classifier on MNIST, convolution visualisation

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build          # 33 tests, all green
./build/xor_demo                # MLP learns XOR → 100% test accuracy
./build/spiral_demo             # MLP learns two spirals → ~98% test accuracy
./build/cnn_demo                # CNN learns circles-vs-squares → 99.25% test accuracy
./build/mnist_demo              # MLP on MNIST → 97.3% test accuracy (needs data/)
./build/benchmark               # op throughput numbers

MNIST data (gitignored, one-time download):

```bash
./scripts/download_mnist.sh
```
```

## Benchmarks

Release build, fwd+bwd per iteration:

| op | workload | ms/iter |
|---|---|---|
| matmul | [512,1024] @ [1024,512] | 160.6 |
| conv2d 3×3 | 32 imgs, 16→32 ch, 28×28 | 473.2 |
| softmax_rows | [512,1024] | 10.4 |
| layernorm | [64,512] | 0.37 |
| full CNN step | 2 conv + 2 pool + fc, batch 64 | 87.5 |

Naive direct kernels (no BLAS, no im2col) — fast enough to train the demos
in seconds, and the code is the point.

## Quick start

```cpp
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/serialization.hpp"
#include "nn/trainer.hpp"
#include "nn/data.hpp"

nn::Sequential model;
model.add<nn::Linear>(2, 16);
model.add<nn::LayerNorm>(16);
model.add<nn::ReLU>();
model.add<nn::Linear>(16, 2);

nn::Adam opt(model.parameters(), 0.01f);
nn::CosineAnnealingLR sched(opt, /*t_max=*/200);
nn::Dataset train = nn::make_xor(400);
nn::DataLoader loader(train, 32);
nn::TrainConfig cfg{.epochs = 200};
nn::train_classifier(model, loader, opt, nn::cross_entropy, cfg);
sched.step();                       // once per epoch
nn::save_model(model, "xor.nns1");  // checkpoint
```

CNN example (from `examples/cnn_demo.cpp`):

```cpp
nn::Conv2d c1(1, 8, 3, 1, 1);       // [N,1,28,28] -> [N,8,28,28]
nn::MaxPool2d p1(2);                // -> [N,8,14,14]
nn::Var h = p1.forward(nn::relu(c1.forward(nn::reshape_v(x, {n,1,28,28}))));
```

## Project layout

```
include/nn/     public headers (tensor, autograd, layers, conv, optimizers,
                schedulers, serialization, data, trainer)
src/            implementations
tests/          gradient checks + unit tests (CTest)
examples/       xor_demo, spiral_demo, cnn_demo, benchmark
```

## License

MIT — see [LICENSE](LICENSE).
