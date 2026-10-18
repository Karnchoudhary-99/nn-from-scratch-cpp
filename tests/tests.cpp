// nn_tests: unit tests for tensor ops, numeric gradient checks for every
// differentiable op, and an end-to-end training convergence test.
// Zero dependencies — a 30-line harness.
#include "nn/autograd.hpp"
#include "nn/data.hpp"
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/tensor.hpp"
#include "nn/trainer.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int passed = 0, failed = 0;
std::string current;

#define TEST(name)                                   \
    static void test_##name();                       \
    static bool reg_##name = (register_test(#name, test_##name), true); \
    static void test_##name()

std::vector<std::pair<std::string, void (*)()>>& registry() {
    static std::vector<std::pair<std::string, void (*)()>> r;
    return r;
}
void register_test(const char* name, void (*fn)()) { registry().push_back({name, fn}); }

#define REQUIRE(cond)                                                      \
    do {                                                                   \
        if (!(cond))                                                        \
            throw std::runtime_error(std::string("REQUIRE failed: ") + #cond \
                                     + " at line " + std::to_string(__LINE__)); \
    } while (0)

#define REQUIRE_NEAR(a, b, tol)                                            \
    do {                                                                   \
        double _a = (a), _b = (b);                                          \
        if (std::fabs(_a - _b) > (tol))                                    \
            throw std::runtime_error("REQUIRE_NEAR failed: " + std::to_string(_a) \
                                     + " vs " + std::to_string(_b)          \
                                     + " at line " + std::to_string(__LINE__)); \
    } while (0)

// --- tensor tests --------------------------------------------------------

TEST(tensor_shapes_and_indexing) {
    nn::Tensor t({2, 3});
    REQUIRE(t.numel() == 6);
    REQUIRE(t.dim(0) == 2 && t.dim(1) == 3);
    t.at({1, 2}) = 5.0f;
    REQUIRE_NEAR(t.at({1, 2}), 5.0, 1e-9);
    REQUIRE_NEAR(t[5], 5.0, 1e-9);
}

TEST(tensor_broadcast_add) {
    nn::Tensor a = nn::Tensor::full({2, 3}, 1.0f);
    nn::Tensor b = nn::Tensor::full({3}, 2.0f);
    nn::Tensor c = nn::add(a, b);
    REQUIRE(c.same_shape(nn::Tensor({2, 3})));
    for (int i = 0; i < 6; ++i) REQUIRE_NEAR(c[i], 3.0, 1e-6);
    // scalar broadcast
    nn::Tensor d = nn::add_scalar(a, 10.0f);
    REQUIRE_NEAR(d[0], 11.0, 1e-6);
}

TEST(tensor_matmul) {
    nn::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    nn::Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    nn::Tensor c = nn::matmul(a, b);
    // [58, 64; 139, 154]
    REQUIRE_NEAR(c.at({0, 0}), 58.0, 1e-4);
    REQUIRE_NEAR(c.at({0, 1}), 64.0, 1e-4);
    REQUIRE_NEAR(c.at({1, 0}), 139.0, 1e-4);
    REQUIRE_NEAR(c.at({1, 1}), 154.0, 1e-4);
}

TEST(tensor_transpose) {
    nn::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    nn::Tensor t = a.transpose2d();
    REQUIRE(t.dim(0) == 3 && t.dim(1) == 2);
    REQUIRE_NEAR(t.at({2, 1}), 6.0, 1e-6);
    REQUIRE_NEAR(t.at({0, 1}), 4.0, 1e-6);
}

TEST(tensor_softmax_rows_sum_to_one) {
    nn::Tensor a = nn::Tensor::randn({8, 5});
    nn::Tensor s = nn::softmax_rows(a);
    for (int i = 0; i < 8; ++i) {
        float sum = 0;
        for (int j = 0; j < 5; ++j) sum += s.at({i, j});
        REQUIRE_NEAR(sum, 1.0, 1e-5);
    }
    // log-softmax consistency: exp(log_softmax) == softmax
    nn::Tensor ls = nn::log_softmax_rows(a);
    for (int i = 0; i < 40; ++i) REQUIRE_NEAR(std::exp(ls[i]), s[i], 1e-5);
}

TEST(tensor_unbroadcast) {
    nn::Tensor g = nn::Tensor::full({4, 3}, 1.0f);
    nn::Tensor u = nn::unbroadcast(g, {3});
    REQUIRE(u.dim(0) == 3);
    for (int i = 0; i < 3; ++i) REQUIRE_NEAR(u[i], 4.0, 1e-6);
    nn::Tensor s = nn::unbroadcast(nn::Tensor::full({4, 3}, 2.0f), {});
    REQUIRE_NEAR(s.at({}), 24.0, 1e-5);
}

// --- gradient checking ---------------------------------------------------
// Compares analytic gradients against central finite differences.

void gradcheck(const std::string& name,
               std::function<nn::Var(const std::vector<nn::Var>&)> build,
               std::vector<nn::Tensor> inputs, double tol = 1e-2) {
    const double eps = 1e-3;
    std::vector<nn::Var> vars;
    for (auto& t : inputs) vars.push_back(nn::make_var(t));

    nn::Var out = build(vars);
    out->backward();
    std::vector<nn::Tensor> analytic;
    for (auto& v : vars) analytic.push_back(v->grad);

    for (size_t vi = 0; vi < vars.size(); ++vi) {
        nn::Tensor& t = inputs[vi];
        for (int64_t i = 0; i < t.numel(); ++i) {
            const float orig = t[i];
            t[i] = orig + static_cast<float>(eps);
            std::vector<nn::Var> vp;
            for (auto& tt : inputs) vp.push_back(nn::make_var(tt));
            const float fp = build(vp)->data.at({});
            t[i] = orig - static_cast<float>(eps);
            vp.clear();
            for (auto& tt : inputs) vp.push_back(nn::make_var(tt));
            const float fm = build(vp)->data.at({});
            t[i] = orig;
            const double numeric = (fp - fm) / (2 * eps);
            const double ana = analytic[vi][i];
            if (std::fabs(numeric - ana) > tol &&
                std::fabs(numeric - ana) / (std::fabs(numeric) + std::fabs(ana) + 1e-8) > tol)
                throw std::runtime_error(name + ": grad mismatch at input " +
                                         std::to_string(vi) + " idx " + std::to_string(i) +
                                         " analytic=" + std::to_string(ana) +
                                         " numeric=" + std::to_string(numeric));
        }
    }
}

nn::Tensor small(std::vector<int64_t> shape, float lo = -1.0f, float hi = 1.0f) {
    return nn::Tensor::randu(shape, lo, hi);
}

TEST(grad_add_sub_mul_div) {
    nn::set_seed(1);
    auto a = small({2, 3}), b = small({2, 3});
    gradcheck("add", [](auto v) { return nn::sum_all(nn::add(v[0], v[1])); }, {a, b});
    gradcheck("sub", [](auto v) { return nn::sum_all(nn::sub(v[0], v[1])); }, {a, b});
    gradcheck("mul", [](auto v) { return nn::sum_all(nn::mul(v[0], v[1])); }, {a, b});
    auto b2 = small({2, 3}, 0.5f, 1.5f); // keep divisor away from 0
    gradcheck("div", [](auto v) { return nn::sum_all(nn::div(v[0], v[1])); }, {a, b2});
}

TEST(grad_broadcast) {
    nn::set_seed(2);
    auto a = small({4, 3}), b = small({3});
    gradcheck("broadcast-add", [](auto v) { return nn::sum_all(nn::add(v[0], v[1])); }, {a, b});
    gradcheck("broadcast-mul", [](auto v) { return nn::sum_all(nn::mul(v[0], v[1])); }, {a, b});
}

TEST(grad_matmul) {
    nn::set_seed(3);
    auto a = small({4, 5}), b = small({5, 3});
    gradcheck("matmul", [](auto v) { return nn::sum_all(nn::matmul(v[0], v[1])); }, {a, b});
}

TEST(grad_activations) {
    nn::set_seed(4);
    auto a = small({3, 4});
    auto ar = a; // relu: keep inputs away from the kink at 0
    for (int64_t i = 0; i < ar.numel(); ++i) ar[i] += (ar[i] >= 0 ? 0.5f : -0.5f);
    gradcheck("relu", [](auto v) { return nn::sum_all(nn::relu(v[0])); }, {ar});
    gradcheck("sigmoid", [](auto v) { return nn::sum_all(nn::sigmoid(v[0])); }, {a});
    gradcheck("tanh", [](auto v) { return nn::sum_all(nn::tanh_act(v[0])); }, {a});
}

TEST(grad_reductions) {
    nn::set_seed(5);
    auto a = small({3, 4});
    gradcheck("sum_all", [](auto v) { return nn::sum_all(v[0]); }, {a});
    gradcheck("mean_all", [](auto v) { return nn::mean_all(v[0]); }, {a});
}

TEST(grad_linear) {
    nn::set_seed(6);
    auto x = small({5, 4}), w = small({4, 3}), b = small({3});
    gradcheck("linear",
              [](auto v) { return nn::sum_all(nn::linear(v[0], v[1], v[2])); },
              {x, w, b});
}

TEST(grad_mse) {
    nn::set_seed(7);
    auto p = small({4, 3}), t = small({4, 3});
    gradcheck("mse", [](auto v) { return nn::mse_loss(v[0], v[1]); }, {p, t});
}

TEST(grad_cross_entropy) {
    nn::set_seed(8);
    auto logits = small({6, 4});
    std::vector<int> labels = {0, 3, 1, 2, 0, 3};
    gradcheck("cross_entropy",
              [&](auto v) { return nn::cross_entropy(v[0], labels); },
              {logits}, 2e-2);
}

TEST(grad_softmax) {
    nn::set_seed(9);
    auto a = small({4, 5});
    gradcheck("softmax_rows",
              [](auto v) { return nn::sum_all(nn::softmax_rows_v(v[0])); },
              {a});
}

// --- end-to-end ----------------------------------------------------------

TEST(training_xor_converges) {
    nn::set_seed(42);
    nn::Dataset train = nn::make_xor(200);
    nn::Sequential model;
    model.add<nn::Linear>(2, 8);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(8, 2);
    nn::Adam opt(model.parameters(), 0.02f);
    nn::DataLoader loader(train, 32);
    nn::TrainConfig cfg;
    cfg.epochs = 150;
    cfg.verbose = false;
    nn::History h = nn::train_classifier(model, loader, opt, nn::cross_entropy, cfg);
    REQUIRE(h.acc.back() > 0.99f);
    REQUIRE(h.loss.back() < h.loss.front());
    REQUIRE(nn::accuracy(model, train) > 0.99f);
}

TEST(optimizer_sgd_descends) {
    nn::set_seed(11);
    // Linear regression y = 3x + 1 with plain SGD must reduce MSE.
    nn::Var w = nn::make_var(nn::Tensor::randn({1, 1}, 0.0f, 0.1f));
    nn::Var b = nn::make_var(nn::Tensor::zeros({1}));
    nn::SGD opt({w, b}, 0.1f);
    nn::Tensor xt({4, 1}, {1, 2, 3, 4});
    nn::Tensor yt({4, 1}, {4, 7, 10, 13}); // y = 3x + 1
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();
        nn::Var p = nn::add(nn::matmul(nn::constant(xt), w), b);
        nn::Var loss = nn::mse_loss(p, nn::constant(yt));
        if (step == 0) first = loss->data.at({});
        loss->backward();
        opt.step();
        last = loss->data.at({});
    }
    REQUIRE(last < first * 0.01f);
    REQUIRE_NEAR(w->data.at({0, 0}), 3.0, 0.15);
    REQUIRE_NEAR(b->data.at({0}), 1.0, 0.15);
}

// --- scheduler tests --------------------------------------------------------

TEST(schedulers_step_lr) {
    nn::Var w = nn::make_var(nn::Tensor::randn({2, 2}));
    nn::SGD opt({w}, 0.1f);
    nn::StepLR sched(opt, 2, 0.5f);
    REQUIRE_NEAR(opt.get_lr(), 0.1, 1e-6);
    sched.step(); // step 1: no decay yet
    REQUIRE_NEAR(opt.get_lr(), 0.1, 1e-6);
    sched.step(); // step 2: 0.1 * 0.5
    REQUIRE_NEAR(opt.get_lr(), 0.05, 1e-6);
    sched.step(); // step 3
    REQUIRE_NEAR(opt.get_lr(), 0.05, 1e-6);
    sched.step(); // step 4: 0.1 * 0.25
    REQUIRE_NEAR(opt.get_lr(), 0.025, 1e-6);
}

TEST(schedulers_cosine_annealing) {
    nn::Var w = nn::make_var(nn::Tensor::randn({2, 2}));
    nn::Adam opt({w}, 0.2f);
    nn::CosineAnnealingLR sched(opt, 10, 0.0f);
    // t=0 -> lr = 0.2 (initial); after 10 steps lr ~= 0
    for (int i = 0; i < 5; ++i) sched.step();
    REQUIRE_NEAR(opt.get_lr(), 0.1, 1e-5); // cos(pi/2) = 0 -> 0.5*(1+0)=0.5
    for (int i = 0; i < 5; ++i) sched.step();
    REQUIRE_NEAR(opt.get_lr(), 0.0, 1e-6);
    sched.step(); // past T_max: stays at lr_min
    REQUIRE_NEAR(opt.get_lr(), 0.0, 1e-6);
}

TEST(schedulers_exponential_lr) {
    nn::Var w = nn::make_var(nn::Tensor::randn({2, 2}));
    nn::SGD opt({w}, 0.1f);
    nn::ExponentialLR sched(opt, 0.9f);
    sched.step();
    REQUIRE_NEAR(opt.get_lr(), 0.09, 1e-6);
    sched.step();
    REQUIRE_NEAR(opt.get_lr(), 0.081, 1e-6);
}

} // namespace

int main() {
    for (auto& [name, fn] : registry()) {
        current = name;
        try {
            fn();
            ++passed;
            std::printf("PASS %s\n", name.c_str());
        } catch (const std::exception& e) {
            ++failed;
            std::printf("FAIL %s: %s\n", name.c_str(), e.what());
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
