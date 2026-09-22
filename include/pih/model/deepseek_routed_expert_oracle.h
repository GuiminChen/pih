#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"

namespace pih {

struct DeepSeekFp4MatrixView final {
  std::span<const std::byte> packed;
  std::span<const std::byte> scale_bits;
};

class DeepSeekRoutedExpertOracle final {
 public:
  static Result<std::vector<BFloat16>> Forward(
      std::span<const BFloat16> input, std::span<const float> route_weights,
      std::uint32_t token_count, std::uint32_t hidden_size,
      std::uint32_t intermediate_size, DeepSeekFp4MatrixView w1,
      DeepSeekFp4MatrixView w2, DeepSeekFp4MatrixView w3);
};

}  // namespace pih
