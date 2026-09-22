#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct DeepSeekFp8Activation final {
  std::vector<std::uint8_t> e4m3_bits;
  std::vector<std::uint8_t> scale_bits;
  std::uint32_t token_count = 0;
  std::uint32_t logical_k = 0;
};

// Independent scalar oracle for the official K128 activation quantizer used by
// FP4 expert GEMMs. Scale bytes are encoded UE8M0 values, not numeric uint8.
class DeepSeekFp8ActivationCodec final {
 public:
  static constexpr std::uint32_t kValuesPerScale = 128;
  static constexpr float kMaxFinite = 448.0F;
  static constexpr float kAmaxFloor = 1.0e-4F;

  static Result<float> DecodeE4m3Fn(std::uint8_t bits);
  static Result<std::uint8_t> EncodeE4m3Fn(float value);
  static Result<DeepSeekFp8Activation> Quantize(
      std::span<const float> input, std::uint32_t token_count,
      std::uint32_t logical_k);
};

}  // namespace pih
