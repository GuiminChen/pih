#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"

namespace pih {

class DeepSeekIndexScoreOracle final {
 public:
  static constexpr std::uint32_t kHeadDim = 128;
  static constexpr std::uint32_t kMaximumHeads = 64;
  static constexpr std::uint32_t kMaximumSlotTile = 4096;

  static Result<std::vector<float>> Evaluate(
      std::span<const BFloat16> query,
      std::span<const BFloat16> index_kv,
      std::span<const float> head_weight,
      std::uint32_t query_count, std::uint32_t head_count,
      std::uint32_t slot_count);
  static Result<std::vector<float>> EvaluatePaged(
      std::span<const BFloat16> query,
      std::span<const BFloat16> physical_index_kv,
      std::span<const float> head_weight,
      std::span<const std::uint32_t> page_slots,
      std::uint32_t query_count, std::uint32_t head_count,
      std::uint32_t slot_base, std::uint32_t slot_count,
      std::uint32_t physical_page_count);
};

}  // namespace pih
