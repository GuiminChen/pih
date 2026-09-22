#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekComponentSha256Launch final {
  std::uintptr_t input = 0;
  std::uint64_t bytes = 0;
  std::uintptr_t output_digest = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
};

Status launch_deepseek_component_sha256(
    DeepSeekComponentSha256Launch launch);

}  // namespace pih
