#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekExpertSwiGluLaunch final {
  static constexpr std::uint32_t kIntermediateSize = 2048;
  static constexpr float kLimit = 10.0F;

  std::uintptr_t gate_bf16 = 0;
  std::uintptr_t up_bf16 = 0;
  std::uintptr_t route_weights_f32 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
};

struct DeepSeekSharedExpertSwiGluLaunch final {
  static constexpr std::uint32_t kIntermediateSize = 2048;
  static constexpr float kLimit = 10.0F;

  std::uintptr_t gate_bf16 = 0;
  std::uintptr_t up_bf16 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
};

Status validate_deepseek_expert_swiglu_launch(
    const DeepSeekExpertSwiGluLaunch& launch);
Status launch_deepseek_expert_swiglu(DeepSeekExpertSwiGluLaunch launch);
Status validate_deepseek_shared_expert_swiglu_launch(
    const DeepSeekSharedExpertSwiGluLaunch& launch);
Status launch_deepseek_shared_expert_swiglu(
    DeepSeekSharedExpertSwiGluLaunch launch);

}  // namespace pih
