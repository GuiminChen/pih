#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

enum class DeepSeekFp8GemmOutputType : std::uint8_t {
  kBf16 = 1,
  kFp32 = 2,
};

struct DeepSeekFp8GemmLaunch final {
  std::uintptr_t activation_e4m3 = 0;
  std::uintptr_t activation_scale_bits = 0;
  std::uintptr_t weight_e4m3 = 0;
  std::uintptr_t weight_scale_bits = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  DeepSeekFp8GemmOutputType output_type =
      DeepSeekFp8GemmOutputType::kBf16;
};

Status validate_deepseek_fp8_gemm_launch(const DeepSeekFp8GemmLaunch& launch);
Status launch_deepseek_fp8_gemm(DeepSeekFp8GemmLaunch launch);

}  // namespace pih
