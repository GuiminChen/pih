#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekExpertAccumulateLaunch final {
  std::uintptr_t expert_output_bf16 = 0;
  std::uintptr_t token_indices = 0;
  std::uintptr_t accumulator_f32 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t route_count = 0;
  std::uint32_t token_count = 0;
};

struct DeepSeekExpertFinalizeLaunch final {
  std::uintptr_t accumulator_f32 = 0;
  std::uintptr_t shared_output_bf16 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
};

Status launch_deepseek_ordered_expert_accumulate(
    DeepSeekExpertAccumulateLaunch launch);
Status launch_deepseek_expert_finalize(DeepSeekExpertFinalizeLaunch launch);

}  // namespace pih
