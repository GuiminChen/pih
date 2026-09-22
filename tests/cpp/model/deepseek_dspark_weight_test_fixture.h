#pragma once

#include <functional>
#include <string>
#include <vector>

#include "pih/model/deepseek_dspark_weight_bindings.h"

namespace pih::test {

using DeepSeekDsparkTestShape = std::vector<std::int64_t>;

inline Result<TensorView> deepseek_dspark_test_tensor(
    std::string_view name, std::vector<std::string>* requested = nullptr,
    bool wrong_stage2_head = false, std::uint64_t generation = 29) {
  if (requested != nullptr) requested->emplace_back(name);
  if (name.size() < 6 || !name.starts_with("mtp.") || name[5] != '.' ||
      name[4] < '0' || name[4] > '2') {
    return Status::InvalidArgument("non-canonical DSpark tensor name");
  }
  const std::string_view suffix = name.substr(6);
  DType dtype = DType::kBFloat16;
  DeepSeekDsparkTestShape shape;
  if (suffix == "attn.wq_a.weight") {
    dtype = DType::kFloat8E4M3; shape = {1024, 4096};
  } else if (suffix == "attn.wq_a.scale") {
    dtype = DType::kFloat8E8M0; shape = {8, 32};
  } else if (suffix == "attn.q_norm.weight") {
    shape = {1024};
  } else if (suffix == "attn.wq_b.weight") {
    dtype = DType::kFloat8E4M3; shape = {32768, 1024};
  } else if (suffix == "attn.wq_b.scale") {
    dtype = DType::kFloat8E8M0; shape = {256, 8};
  } else if (suffix == "attn.wkv.weight") {
    dtype = DType::kFloat8E4M3; shape = {512, 4096};
  } else if (suffix == "attn.wkv.scale") {
    dtype = DType::kFloat8E8M0; shape = {4, 32};
  } else if (suffix == "attn.kv_norm.weight") {
    shape = {512};
  } else if (suffix == "attn.attn_sink") {
    dtype = DType::kFloat32; shape = {64};
  } else if (suffix == "attn.wo_a.weight") {
    dtype = DType::kFloat8E4M3; shape = {8192, 4096};
  } else if (suffix == "attn.wo_a.scale") {
    dtype = DType::kFloat8E8M0; shape = {64, 32};
  } else if (suffix == "attn.wo_b.weight") {
    dtype = DType::kFloat8E4M3; shape = {4096, 8192};
  } else if (suffix == "attn.wo_b.scale") {
    dtype = DType::kFloat8E8M0; shape = {32, 64};
  } else if (suffix == "attn_norm.weight" ||
             suffix == "ffn_norm.weight") {
    shape = {4096};
  } else if (suffix == "hc_attn_fn" || suffix == "hc_ffn_fn") {
    dtype = DType::kFloat32; shape = {24, 16384};
  } else if (suffix == "hc_attn_scale" || suffix == "hc_ffn_scale") {
    dtype = DType::kFloat32; shape = {3};
  } else if (suffix == "hc_attn_base" || suffix == "hc_ffn_base") {
    dtype = DType::kFloat32; shape = {24};
  } else if (suffix == "ffn.gate.weight") {
    shape = {256, 4096};
  } else if (suffix == "ffn.gate.bias") {
    dtype = DType::kFloat32; shape = {256};
  } else if (suffix.starts_with("ffn.shared_experts.")) {
    const bool scale = suffix.ends_with(".scale");
    const bool w2 = suffix.find(".w2.") != std::string_view::npos;
    dtype = scale ? DType::kFloat8E8M0 : DType::kInt8;
    shape = scale ? DeepSeekDsparkTestShape{w2 ? 4096 : 2048,
                                           w2 ? 64 : 128}
                  : DeepSeekDsparkTestShape{w2 ? 4096 : 2048,
                                           w2 ? 1024 : 2048};
  } else if (suffix == "main_proj.weight") {
    dtype = DType::kFloat8E4M3; shape = {4096, 12288};
  } else if (suffix == "main_proj.scale") {
    dtype = DType::kFloat8E8M0; shape = {32, 96};
  } else if (suffix == "main_norm.weight" || suffix == "norm.weight") {
    shape = {4096};
  } else if (suffix == "markov_head.markov_w1.weight" ||
             suffix == "markov_head.markov_w2.weight") {
    shape = {129280, 256};
  } else if (suffix == "confidence_head.proj.weight") {
    shape = wrong_stage2_head ? DeepSeekDsparkTestShape{4352, 1}
                              : DeepSeekDsparkTestShape{1, 4352};
  } else if (suffix == "hc_head_fn") {
    dtype = DType::kFloat32; shape = {4, 16384};
  } else if (suffix == "hc_head_scale") {
    dtype = DType::kFloat32; shape = {1};
  } else if (suffix == "hc_head_base") {
    dtype = DType::kFloat32; shape = {4};
  } else {
    return Status::InvalidArgument("missing DSpark tensor");
  }
  const auto address = static_cast<std::uintptr_t>(
      0x10000U + (std::hash<std::string_view>{}(name) & 0x0fffffffU));
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
                            Device::Create(DeviceType::kCuda, 0).value(),
                            generation);
}

inline DeepSeekStagePlan deepseek_dspark_test_stage() {
  return {1, {21, 42}, false, true, true};
}

inline DeepSeekDsparkWeightBindings deepseek_dspark_test_weights(
    std::uint64_t generation = 29) {
  return DeepSeekDsparkWeightBindings::Resolve(
      deepseek_dspark_test_stage(), [generation](std::string_view name) {
        return deepseek_dspark_test_tensor(name, nullptr, false, generation);
      }).value();
}

}  // namespace pih::test
