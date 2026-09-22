#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekRopeTableLaunch final {
  std::uintptr_t output_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t position_count = 0;
  std::uint32_t rope_dimension = 64;
  double theta = 10000.0;
  double scaling_factor = 1.0;
  std::uint32_t original_maximum_positions = 65536;
  std::uint32_t beta_fast = 32;
  std::uint32_t beta_slow = 1;
  bool yarn = false;
};

Status validate_deepseek_rope_table_launch(
    const DeepSeekRopeTableLaunch& launch);
Status launch_deepseek_rope_table(DeepSeekRopeTableLaunch launch);

class DeepSeekRopeTableOperations {
 public:
  virtual ~DeepSeekRopeTableOperations() = default;
  virtual Status initialize(DeepSeekRopeTableLaunch launch) = 0;
  virtual Status synchronize(std::uintptr_t stream) = 0;
};

}  // namespace pih
