// benchmark: forward+backward throughput of the core ops (no training).
#include "nn/autograd.hpp"
#include "nn/conv.hpp"
#include "nn/layers.hpp"

#include <chrono>
#include <cstdio>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Time `fn` over `iters` iterations; returns median-ish mean ms/iter.
template <typename F> double bench(const char* name, int iters, F&& fn) {
    fn(); // warmup
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) fn();
    const double ms = (now_ms() - t0) / iters;
    std::printf("%-34s %8.2f ms/iter\n", name, ms);
    return ms;
}

} // namespace

int main() {
    nn::set_seed(1);
    std::printf("--- nn-from-scratch-cpp benchmark (release build) ---\n");

    {
        nn::Var a = nn::make_var(nn::Tensor::randn({512, 1024}));
        nn::Var b = nn::make_var(nn::Tensor::randn({1024, 512}));
        bench("matmul [512,1024]@[1024,512] fwd+bwd", 20, [&] {
            nn::Var c = nn::matmul(a, b);
            nn::sum_all(c)->backward();
            a->zero_grad();
            b->zero_grad();
        });
    }
    {
        // Typical conv workload: batch 32, 16ch 28x28 in, 32ch 3x3 out.
        nn::Var x = nn::make_var(nn::Tensor::randn({32, 16, 28, 28}));
        nn::Var w = nn::make_var(nn::Tensor::randn({32, 16, 3, 3}));
        nn::Var bias = nn::make_var(nn::Tensor::zeros({32}));
        bench("conv2d 3x3 (32b,16->32ch,28x28) fwd+bwd", 5, [&] {
            nn::Var y = nn::conv2d_v(x, w, bias, 1, 1);
            nn::sum_all(y)->backward();
            x->zero_grad();
            w->zero_grad();
            bias->zero_grad();
        });
    }
    {
        nn::Var a = nn::make_var(nn::Tensor::randn({512, 1024}));
        bench("softmax_rows [512,1024] fwd+bwd", 20, [&] {
            nn::Var s = nn::softmax_rows_v(a);
            nn::sum_all(s)->backward();
            a->zero_grad();
        });
    }
    {
        nn::Var x = nn::make_var(nn::Tensor::randn({64, 512}));
        nn::Var g = nn::make_var(nn::Tensor::ones({512}));
        nn::Var b = nn::make_var(nn::Tensor::zeros({512}));
        bench("layernorm [64,512] fwd+bwd", 50, [&] {
            nn::Var y = nn::layernorm_v(x, g, b);
            nn::sum_all(y)->backward();
            x->zero_grad();
            g->zero_grad();
            b->zero_grad();
        });
    }
    {
        // End-to-end: one CNN forward+backward step (the cnn_demo workload).
        nn::set_seed(7);
        nn::Conv2d c1(1, 8, 3, 1, 1), c2(8, 16, 3, 1, 1);
        nn::Linear fc(16 * 7 * 7, 2);
        nn::Var x = nn::make_var(nn::Tensor::randn({64, 784}));
        std::vector<int> y(64, 0);
        bench("cnn step: 2xconv+pool+fc [64x784] fwd+bwd", 10, [&] {
            nn::Var h = nn::reshape_v(x, {64, 1, 28, 28});
            h = nn::maxpool2d_v(nn::relu(c1.forward(h)), 2, 2);
            h = nn::maxpool2d_v(nn::relu(c2.forward(h)), 2, 2);
            h = nn::reshape_v(h, {64, 16 * 7 * 7});
            nn::Var logits = fc.forward(h);
            nn::cross_entropy(logits, y)->backward();
            for (auto p : {c1.weight, c1.bias, c2.weight, c2.bias, fc.weight,
                           fc.bias})
                p->zero_grad();
            x->zero_grad();
        });
    }
    return 0;
}
