// Serialization: save/load a Module's learnable parameters to a portable
// binary checkpoint file ("NNS1" format, little-endian).
//
// Layout: magic "NNS1" | u32 param_count | per param: u32 name_len, name bytes,
// u8 ndim, ndim x u64 dims, numel x float32 data.
// load_model requires the same architecture (same names and shapes).
#pragma once

#include "nn/layers.hpp"

#include <string>

namespace nn {

void save_model(Module& model, const std::string& path);
void load_model(Module& model, const std::string& path);

} // namespace nn
