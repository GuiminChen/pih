#include "pih/backend/cuda/deepseek_rope_table.h"

#include <cmath>

namespace pih {

Status validate_deepseek_rope_table_launch(
    const DeepSeekRopeTableLaunch& value) {
  if (value.output_f32 == 0 || value.error_flag_u32 == 0 ||
      value.stream == 0 || value.position_count == 0 ||
      value.position_count > 1048576 || value.rope_dimension != 64 ||
      !std::isfinite(value.theta) || value.theta != 10000.0 ||
      !std::isfinite(value.scaling_factor) ||
      (value.yarn ? value.scaling_factor != 16.0
                  : value.scaling_factor != 1.0) ||
      value.original_maximum_positions != 65536 ||
      value.beta_fast != 32 || value.beta_slow != 1) {
    return Status::InvalidArgument("DeepSeek RoPE table launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
