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

- [ ] `Tensor`: N-dimensional array, row-major storage, broadcasting, matmul
- [ ] Autograd: computation graph, reverse-mode differentiation
- [ ] Layers: Linear, ReLU/Sigmoid/Tanh/Softmax, Dropout, BatchNorm
- [ ] Losses: MSE, cross-entropy (with log-softmax fusion)
- [ ] Optimizers: SGD, SGD+momentum, Adam
- [ ] Data: MNIST loader (IDX format, no dependencies)
- [ ] Training loop: mini-batch, shuffling, validation split, checkpoints
- [ ] Demos: XOR, spiral classification, MNIST digit classifier
- [ ] Benchmarks: forward/backward throughput vs. a Python baseline
- [ ] Tests: gradient checking (numeric vs. analytic), layer unit tests

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/mnist_demo
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
