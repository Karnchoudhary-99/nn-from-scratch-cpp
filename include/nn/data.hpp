// Data: synthetic 2-D classification datasets and a shuffling DataLoader.
// (MNIST IDX loader lands in a later session.)
#pragma once

#include "nn/tensor.hpp"

#include <cstddef>
#include <vector>

namespace nn {

struct Dataset {
    std::vector<Tensor> xs; // each [D]
    std::vector<int> ys;
    size_t size() const { return xs.size(); }
    int num_classes = 2;
};

// Four noisy XOR clusters around the unit-square corners.
Dataset make_xor(size_t n, float noise = 0.08f);
// Two interleaving half-moons.
Dataset make_moons(size_t n, float noise = 0.08f);
// Two interleaving spirals.
Dataset make_spirals(size_t n, float noise = 0.05f);
// Gaussian blobs, one per class.
Dataset make_blobs(size_t n, int classes = 3, float spread = 0.6f);
// 28x28 images: filled circles (class 0) vs filled squares (class 1),
// random position/scale/rotation, flattened to [784]. Feeds the CNN demo
// without any downloads.
Dataset make_shapes(size_t n, int size = 28, float noise = 0.08f);

struct Batch {
    Tensor X;              // [B, D]
    std::vector<int> y;    // [B]
};

class DataLoader {
public:
    DataLoader(const Dataset& ds, size_t batch_size, bool shuffle = true);

    // Materialise one epoch of batches (reshuffles when shuffle=true).
    std::vector<Batch> batches();

private:
    const Dataset& ds_;
    size_t batch_size_;
    bool shuffle_;
};

} // namespace nn
