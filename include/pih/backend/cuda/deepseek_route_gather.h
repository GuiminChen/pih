#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekRouteGatherLaunch final {
  static constexpr std::uint32_t kHiddenSize = 4096;

  std::uintptr_t source_hidden_bf16 = 0;
  std::uintptr_t token_indices_u32 = 0;
  std::uintptr_t route_hidden_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t route_count = 0;
  std::uint32_t packed_token_count = 0;
};

Status validate_deepseek_route_gather_launch(
    const DeepSeekRouteGatherLaunch& launch);
Status launch_deepseek_route_gather(DeepSeekRouteGatherLaunch launch);

}  // namespace pih
