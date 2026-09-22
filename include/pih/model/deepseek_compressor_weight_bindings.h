#pragma once

#include <functional>
#include <string_view>

#include "pih/model/deepseek_attention_layer_coordinator.h"
#include "pih/model/deepseek_resident_weight_arena.h"

namespace pih {

struct DeepSeekCompressorWeightBindings final {
  std::uintptr_t main_wkv_bf16 = 0;
  std::uintptr_t main_wgate_bf16 = 0;
  std::uintptr_t main_ape_f32 = 0;
  std::uintptr_t main_norm_bf16 = 0;
  std::uintptr_t indexer_wq_b_e4m3 = 0;
  std::uintptr_t indexer_wq_b_scale_bits = 0;
  std::uintptr_t indexer_weights_proj_bf16 = 0;
  std::uintptr_t indexer_wkv_bf16 = 0;
  std::uintptr_t indexer_wgate_bf16 = 0;
  std::uintptr_t indexer_ape_f32 = 0;
  std::uintptr_t indexer_norm_bf16 = 0;
  std::uint64_t generation = 0;

  using Resolver = std::function<Result<TensorView>(std::string_view)>;
  static Result<DeepSeekCompressorWeightBindings> Resolve(
      std::uint32_t layer, DeepSeekCompressedAttentionKind kind,
      const Resolver& resolver);
  static Result<DeepSeekCompressorWeightBindings> Resolve(
      std::uint32_t layer, DeepSeekCompressedAttentionKind kind,
      const DeepSeekResidentWeightArena& arena);
};

}  // namespace pih
