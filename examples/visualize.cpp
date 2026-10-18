// visualize: peek inside the trained MNIST CNN. Dumps plain-PPM images of
// conv1's learned filters and of the conv1/conv2 feature maps for a few
// sample digits, so you can see what the network actually learns.
// Usage: ./build/visualize [output_dir]  (default: ./vis)
// Loads /tmp/mnist_cnn.nns1 if present (train it with mnist_cnn first);
// otherwise trains one quick epoch itself.
// Convert the PPMs to PNGs with:  python3 scripts/render_vis.py vis docs
#include "nn/autograd.hpp"
#include "nn/data.hpp"
#include "nn/optimizers.hpp"
#include "nn/serialization.hpp"
#include "nn/trainer.hpp"
#include "mnist_cnn_model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Write a grayscale PPM (P5) from a [H x W] float plane, min-max normalised.
void write_ppm(const std::string& path, const std::vector<float>& px, int h,
               int w, int scale) {
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << w * scale << " " << h * scale << "\n255\n";
    float lo = px[0], hi = px[0];
    for (float v : px) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float span = hi - lo < 1e-6f ? 1.0f : hi - lo;
    for (int y = 0; y < h; ++y)
        for (int sy = 0; sy < scale; ++sy)
            for (int x = 0; x < w; ++x) {
                const float v = px[static_cast<size_t>(y) * w + x];
                const char byte = static_cast<char>(
                    255.0f * (v - lo) / span + 0.5f);
                for (int sx = 0; sx < scale; ++sx) f.put(byte);
            }
}

// Tile a [C, H, W] tensor's channels into one flat pixel grid.
std::vector<float> tile_maps(const nn::Tensor& t, int cols, int& out_h,
                             int& out_w) {
    const int c = static_cast<int>(t.dim(0)), h = static_cast<int>(t.dim(1)),
              w = static_cast<int>(t.dim(2));
    const int rows = (c + cols - 1) / cols;
    out_h = rows * h;
    out_w = cols * w;
    std::vector<float> px(static_cast<size_t>(out_h) * out_w, 0.0f);
    for (int ch = 0; ch < c; ++ch) {
        const int r = ch / cols, cc = ch % cols;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                px[static_cast<size_t>(r * h + y) * out_w + cc * w + x] =
                    t.at({ch, y, x});
    }
    return px;
}

} // namespace

int main(int argc, char** argv) {
    nn::set_seed(2026);
    const std::string out = argc > 1 ? argv[1] : "./vis";
    if (std::system(("mkdir -p " + out).c_str()) != 0) {
        std::fprintf(stderr, "warning: could not create dir %s\n", out.c_str());
    }

    nn::Dataset test;
    try {
        test = nn::load_mnist_idx("data/t10k-images-idx3-ubyte",
                                  "data/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::printf("MNIST data not found in data/ (%s)\n", e.what());
        return 2;
    }

    nn::MnistCNN model;
    bool loaded = false;
    try {
        nn::load_model(model, "/tmp/mnist_cnn.nns1");
        loaded = true;
        std::printf("loaded checkpoint /tmp/mnist_cnn.nns1\n");
    } catch (const std::exception&) {
        std::printf("no checkpoint found — training one quick epoch...\n");
        nn::Dataset train = nn::load_mnist_idx("data/train-images-idx3-ubyte",
                                               "data/train-labels-idx1-ubyte");
        train.xs.resize(12000);
        train.ys.resize(12000);
        nn::DataLoader loader(train, 64, true);
        nn::Adam opt(model.parameters(), 1e-3f);
        nn::TrainConfig cfg;
        cfg.epochs = 1;
        cfg.log_every = 1;
        nn::train_classifier(
            model, loader, opt,
            [](nn::Var logits, const std::vector<int>& y) {
                return nn::cross_entropy(logits, y);
            },
            cfg);
    }
    model.eval();

    // 1. conv1 filters: weight [8, 1, 5, 5] -> 8 tiles, 5x5 each.
    {
        const nn::Tensor& w = model.c1.weight->data; // [8,1,5,5]
        std::vector<float> grid;
        int gh = 0, gw = 0;
        const int cols = 4;
        const int rows = 2, fh = 5, fw = 5;
        gh = rows * fh;
        gw = cols * fw;
        grid.assign(static_cast<size_t>(gh) * gw, 0.0f);
        for (int f = 0; f < 8; ++f)
            for (int y = 0; y < fh; ++y)
                for (int x = 0; x < fw; ++x)
                    grid[static_cast<size_t>((f / cols) * fh + y) * gw +
                         (f % cols) * fw + x] =
                        w.at({f, 0, y, x});
        write_ppm(out + "/conv1_filters.ppm", grid, gh, gw, 12);
        std::printf("wrote %s/conv1_filters.ppm\n", out.c_str());
    }

    // 2. Feature maps for a few sample digits.
    for (int s = 0; s < 3; ++s) {
        const int idx = s * 37; // arbitrary spread across the test set
        nn::Tensor img({1, 1, 28, 28});
        std::vector<float> input_px(784);
        for (int64_t j = 0; j < 784; ++j) {
            const float v = test.xs[idx].at({j});
            img.at({0, 0, j / 28, j % 28}) = v;
            input_px[static_cast<size_t>(j)] = v;
        }
        write_ppm(out + "/digit_" + std::to_string(s) + "_input.ppm", input_px,
                  28, 28, 6);

        nn::Var x = nn::constant(img);
        nn::Var a1 = model.relu.forward(model.c1.forward(x)); // [1,8,28,28]
        nn::Var a2 =
            model.relu.forward(model.c2.forward(model.p1.forward(a1)));

        nn::Tensor m1 = a1->data.reshape({8, 28, 28});
        nn::Tensor m2 = a2->data.reshape({16, 14, 14});
        int h1, w1, h2, w2;
        auto px1 = tile_maps(m1, 4, h1, w1);
        auto px2 = tile_maps(m2, 4, h2, w2);
        write_ppm(out + "/digit_" + std::to_string(s) + "_conv1_maps.ppm", px1,
                  h1, w1, 4);
        write_ppm(out + "/digit_" + std::to_string(s) + "_conv2_maps.ppm", px2,
                  h2, w2, 8);

        nn::Var logits = model.forward(x);
        const int pred = static_cast<int>(nn::argmax_row(logits->data, 0));
        std::printf("sample %d: true=%d predicted=%d (checkpoint %s)\n", s,
                    test.ys[idx], pred, loaded ? "loaded" : "quick-trained");
    }
    std::printf("done — convert with: python3 scripts/render_vis.py %s docs\n",
                out.c_str());
    return 0;
}
