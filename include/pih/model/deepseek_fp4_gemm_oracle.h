#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/model/deepseek_fp8_activation_codec.h"

namespace pih {

// Scalar reference for C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T. It follows the
// official K32 accumulation order and applies the K128 activation scale and K32
// weight scale to each subblock before adding it to the FP32 output.
class DeepSeekFp4GemmOracle final {
 public:
  static Result<std::vector<float>> Multiply(
      const DeepSeekFp8Activation& activation,
      std::span<const std::byte> packed_weights,
      std::span<const std::byte> weight_scale_bits,
      std::uint32_t output_size);
};

}  // namespace pih
