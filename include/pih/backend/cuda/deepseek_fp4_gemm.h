#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekFp4GemmLaunch final {
  std::uintptr_t activation_e4m3 = 0;
  std::uintptr_t activation_scale_bits = 0;
  std::uintptr_t packed_weight = 0;
  std::uintptr_t weight_scale_bits = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
};

Status validate_deepseek_fp4_gemm_launch(
    const DeepSeekFp4GemmLaunch& launch);
Status launch_deepseek_fp4_gemm(DeepSeekFp4GemmLaunch launch);

}  // namespace pih
