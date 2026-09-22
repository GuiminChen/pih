#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekRouterBf16GemmLaunch final {
  std::uintptr_t input_bf16 = 0;
  std::uintptr_t weight_bf16 = 0;
  std::uintptr_t scores_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t expert_count = 0;
  std::uint32_t hidden_size = 0;
};

Status validate_deepseek_router_bf16_gemm_launch(
    const DeepSeekRouterBf16GemmLaunch& launch);
Status launch_deepseek_router_bf16_gemm(
    DeepSeekRouterBf16GemmLaunch launch);

}  // namespace pih
