// Synthetic dataset generators + DataLoader.
#include "nn/data.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace nn {
namespace {

std::mt19937& data_rng() {
    static std::mt19937 gen(7);
    return gen;
}

float gauss(float std) {
    std::normal_distribution<float> d(0.0f, std);
    return d(data_rng());
}

Dataset build(std::vector<std::pair<float, float>> pts, std::vector<int> labels,
              int classes) {
    Dataset ds;
    ds.num_classes = classes;
    ds.xs.reserve(pts.size());
    for (auto [x, y] : pts) ds.xs.push_back(Tensor({2}, {x, y}));
    ds.ys = std::move(labels);
    return ds;
}

} // namespace

Dataset make_xor(size_t n, float noise) {
    std::vector<std::pair<float, float>> pts;
    std::vector<int> labels;
    pts.reserve(n);
    labels.reserve(n);
    std::uniform_int_distribution<int> corner(0, 3);
    for (size_t i = 0; i < n; ++i) {
        const int c = corner(data_rng());
        const float bx = (c & 1) ? 1.0f : 0.0f;
        const float by = (c & 2) ? 1.0f : 0.0f;
        pts.emplace_back(bx + gauss(noise), by + gauss(noise));
        labels.push_back((c == 1 || c == 2) ? 1 : 0);
    }
    return build(pts, labels, 2);
}

Dataset make_moons(size_t n, float noise) {
    std::vector<std::pair<float, float>> pts;
    std::vector<int> labels;
    pts.reserve(n);
    labels.reserve(n);
    std::uniform_real_distribution<float> ang(0.0f, std::numbers::pi_v<float>);
    for (size_t i = 0; i < n; ++i) {
        const float t = ang(data_rng());
        if (i % 2 == 0) {
            pts.emplace_back(std::cos(t) + gauss(noise), std::sin(t) + gauss(noise));
            labels.push_back(0);
        } else {
            pts.emplace_back(1.0f - std::cos(t) + gauss(noise),
                             0.5f - std::sin(t) + gauss(noise));
            labels.push_back(1);
        }
    }
    return build(pts, labels, 2);
}

Dataset make_spirals(size_t n, float noise) {
    std::vector<std::pair<float, float>> pts;
    std::vector<int> labels;
    pts.reserve(n);
    labels.reserve(n);
    std::uniform_real_distribution<float> r(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        const float rad = r(data_rng());
        const float t = rad * 3.5f * std::numbers::pi_v<float> / 2.0f;
        const float sgn = (i % 2 == 0) ? 1.0f : -1.0f;
        pts.emplace_back(sgn * rad * std::cos(t) + gauss(noise),
                         sgn * rad * std::sin(t) + gauss(noise));
        labels.push_back(i % 2 == 0 ? 0 : 1);
    }
    return build(pts, labels, 2);
}

Dataset make_blobs(size_t n, int classes, float spread) {
    std::vector<std::pair<float, float>> pts;
    std::vector<int> labels;
    pts.reserve(n);
    labels.reserve(n);
    std::uniform_int_distribution<int> cls(0, classes - 1);
    for (size_t i = 0; i < n; ++i) {
        const int c = cls(data_rng());
        const float ang = 2.0f * std::numbers::pi_v<float> * c / classes;
        pts.emplace_back(2.0f * std::cos(ang) + gauss(spread),
                         2.0f * std::sin(ang) + gauss(spread));
        labels.push_back(c);
    }
    return build(pts, labels, classes);
}

DataLoader::DataLoader(const Dataset& ds, size_t batch_size, bool shuffle)
    : ds_(ds), batch_size_(batch_size), shuffle_(shuffle) {}

std::vector<Batch> DataLoader::batches() {
    std::vector<size_t> idx(ds_.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    if (shuffle_) std::shuffle(idx.begin(), idx.end(), data_rng());

    std::vector<Batch> out;
    const int64_t d = ds_.xs[0].dim(0);
    for (size_t s = 0; s < idx.size(); s += batch_size_) {
        const size_t e = std::min(s + batch_size_, idx.size());
        Batch b;
        b.X = Tensor({static_cast<int64_t>(e - s), d});
        b.y.reserve(e - s);
        for (size_t k = s; k < e; ++k) {
            const size_t j = idx[k];
            for (int64_t f = 0; f < d; ++f) b.X.at({static_cast<int64_t>(k - s), f}) = ds_.xs[j][f];
            b.y.push_back(ds_.ys[j]);
        }
        out.push_back(std::move(b));
    }
    return out;
}

Dataset make_shapes(size_t n, int size, float noise) {
    Dataset ds;
    ds.num_classes = 2;
    ds.xs.reserve(n);
    ds.ys.reserve(n);
    std::uniform_real_distribution<float> pos(4.0f, size - 5.0f);
    std::uniform_real_distribution<float> radius(size * 0.16f, size * 0.30f);
    std::uniform_int_distribution<int> cls(0, 1);
    std::normal_distribution<float> nz(0.0f, noise);

    for (size_t i = 0; i < n; ++i) {
        const int c = cls(data_rng());
        const float cx = pos(data_rng()), cy = pos(data_rng());
        const float r = radius(data_rng());
        Tensor img({size * size});
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float dx = x - cx, dy = y - cy;
                bool inside = false;
                if (c == 0) {
                    inside = dx * dx + dy * dy <= r * r; // circle
                } else {
                    // square rotated 45 deg (diamond) for extra challenge
                    inside = std::fabs(dx + dy) + std::fabs(dx - dy) <=
                             2.0f * r * 0.9f;
                }
                float v = inside ? 1.0f : 0.0f;
                v = std::clamp(v + nz(data_rng()), 0.0f, 1.0f);
                img[y * size + x] = v;
            }
        }
        ds.xs.push_back(std::move(img));
        ds.ys.push_back(c);
    }
    return ds;
}

} // namespace nn
