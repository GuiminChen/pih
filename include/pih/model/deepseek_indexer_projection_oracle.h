#pragma once

#include <span>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"

namespace pih {

struct DeepSeekIndexerProjectionOracleOutput final {
  std::vector<BFloat16> query;
  std::vector<float> head_weight;
};

class DeepSeekIndexerProjectionOracle final {
 public:
  static Result<DeepSeekIndexerProjectionOracleOutput> Evaluate(
      std::span<const BFloat16> qr,
      std::span<const BFloat16> hidden,
      std::span<const BFloat16> wq_b,
      std::span<const BFloat16> weights_proj,
      std::span<const float> frequencies,
      std::span<const std::uint32_t> positions,
      std::uint32_t table_position_count);
};

}  // namespace pih
