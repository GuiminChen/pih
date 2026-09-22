#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekFp8ActivationQuantLaunch final {
  static constexpr std::uint32_t kGroupSize = 128;

  std::uintptr_t input_bf16 = 0;
  std::uintptr_t output_e4m3 = 0;
  std::uintptr_t scale_bits = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t logical_k = 0;
};

Status validate_deepseek_fp8_activation_quant_launch(
    const DeepSeekFp8ActivationQuantLaunch& launch);
Status launch_deepseek_fp8_activation_quant(
    DeepSeekFp8ActivationQuantLaunch launch);

}  // namespace pih
