#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/deepseek_fp8_activation_codec.h"

namespace pih {

class DeepSeekFp8GemmOracle final {
 public:
  static Result<std::vector<float>> Multiply(
      const DeepSeekFp8Activation& activation,
      std::span<const std::byte> weight_e4m3,
      std::span<const std::byte> weight_scale_bits,
      std::uint32_t output_size);
};

}  // namespace pih
