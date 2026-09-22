#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

// Scalar semantic oracle for the activation between the expert W1/W3 and W2
// projections. GEMM quantization is deliberately outside this class so the
// K128 activation and K32 weight scale contract remains independently testable.
class DeepSeekExpertSwiGluOracle final {
 public:
  static constexpr std::uint32_t kHiddenSize = 4096;
  static constexpr std::uint32_t kIntermediateSize = 2048;
  static constexpr float kFlash0731Limit = 10.0F;

  static Result<std::vector<float>> Apply(
      std::span<const float> gate, std::span<const float> up,
      std::uint32_t token_count, std::uint32_t intermediate_size,
      float limit = kFlash0731Limit);
};

}  // namespace pih
