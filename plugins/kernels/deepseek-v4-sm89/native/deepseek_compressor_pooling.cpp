#include "pih/backend/cuda/deepseek_compressor_pooling.h"

namespace pih {

Status validate_deepseek_compressor_pooling_launch(
    const DeepSeekCompressorPoolingLaunch& launch) {
  if (launch.kv_projection_f32 == 0 || launch.gate_projection_f32 == 0 ||
      launch.ape_row_f32 == 0 || launch.kv_state_f32 == 0 ||
      launch.score_state_f32 == 0 || launch.output_f32 == 0 ||
      launch.error_flag_u32 == 0 || launch.stream == 0 ||
      launch.batch_count == 0 || launch.batch_count > 4096 ||
      (launch.ratio != 4 && launch.ratio != 128) ||
      (launch.head_dim != 128 && launch.head_dim != 512) ||
      launch.absolute_position >= 1048576) {
    return Status::InvalidArgument(
        "DeepSeek compressor pooling launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
