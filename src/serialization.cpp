// Binary checkpoint save/load ("NNS1" format).
#include "nn/serialization.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace nn {
namespace {

constexpr char kMagic[4] = {'N', 'N', 'S', '1'};

void write_u32(std::ofstream& f, uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void write_u64(std::ofstream& f, uint64_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void write_f32(std::ofstream& f, float v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
uint64_t read_u64(std::ifstream& f) {
    uint64_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
float read_f32(std::ifstream& f) {
    float v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

} // namespace

void save_model(Module& model, const std::string& path) {
    auto params = model.named_parameters();
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_model: cannot open " + path);
    f.write(kMagic, 4);
    write_u32(f, static_cast<uint32_t>(params.size()));
    for (auto& [name, p] : params) {
        const Tensor& t = p->data;
        write_u32(f, static_cast<uint32_t>(name.size()));
        f.write(name.data(), static_cast<std::streamsize>(name.size()));
        const auto& sh = t.shape();
        if (sh.size() > 255)
            throw std::runtime_error("save_model: rank > 255 unsupported");
        f.put(static_cast<char>(sh.size()));
        for (auto d : sh) write_u64(f, static_cast<uint64_t>(d));
        for (int64_t i = 0; i < t.numel(); ++i) write_f32(f, t[i]);
    }
}

void load_model(Module& model, const std::string& path) {
    auto params = model.named_parameters();
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_model: cannot open " + path);
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    if (!f || magic[0] != 'N' || magic[1] != 'N' || magic[2] != 'S' ||
        magic[3] != '1')
        throw std::runtime_error("load_model: bad magic in " + path);
    uint32_t count = read_u32(f);
    if (count != params.size())
        throw std::runtime_error("load_model: checkpoint has " +
                                 std::to_string(count) + " params but model has " +
                                 std::to_string(params.size()));
    for (auto& [name, p] : params) {
        uint32_t name_len = read_u32(f);
        std::string on_disk(name_len, '\0');
        f.read(on_disk.data(), name_len);
        if (on_disk != name)
            throw std::runtime_error("load_model: name mismatch (" + on_disk +
                                     " != " + name + ")");
        int ndim = f.get();
        std::vector<int64_t> shape;
        for (int i = 0; i < ndim; ++i)
            shape.push_back(static_cast<int64_t>(read_u64(f)));
        if (shape != p->data.shape())
            throw std::runtime_error("load_model: shape mismatch for " + name);
        Tensor t(shape);
        for (int64_t i = 0; i < t.numel(); ++i) t[i] = read_f32(f);
        p->data = std::move(t);
    }
    if (f.peek() != EOF)
        throw std::runtime_error("load_model: trailing bytes in " + path);
}

} // namespace nn
