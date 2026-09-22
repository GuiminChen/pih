#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"

namespace pih {

class DeepSeekSparseAttentionOracle final {
 public:
  static constexpr std::uint32_t kHeadDim = 512;
  static constexpr std::uint32_t kTileIndices = 64;
  static constexpr std::uint32_t kMaximumHeads = 64;
  static constexpr std::uint32_t kMaximumIndices = 8320;

  static Result<std::vector<BFloat16>> Evaluate(
      std::span<const BFloat16> query,
      std::span<const BFloat16> latent_kv,
      std::span<const float> attention_sink,
      std::span<const std::int32_t> indices,
      std::uint32_t head_count, std::uint32_t kv_count,
      float softmax_scale);
  static Result<std::vector<BFloat16>> EvaluatePaged(
      std::span<const BFloat16> query,
      std::span<const BFloat16> recent_kv,
      std::span<const BFloat16> compressed_kv,
      std::span<const std::uint32_t> page_slots,
      std::span<const float> attention_sink,
      std::span<const std::int32_t> indices,
      std::uint32_t head_count, std::uint32_t kv_count,
      std::int32_t recent_physical_offset,
      std::int32_t compressed_physical_offset,
      std::uint32_t compressed_slot_count,
      std::uint32_t physical_page_count, float softmax_scale);
};

}  // namespace pih
