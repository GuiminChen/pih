#include "pih/backend/cuda/deepseek_dspark_prefill.h"

namespace pih {

Status validate_deepseek_dspark_recent_store_launch(
    const DeepSeekDsparkRecentStoreLaunch& launch) {
  if (launch.source_bf16 == 0 || launch.positions_u32 == 0 ||
      launch.recent_ring_bf16 == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0 || launch.token_count == 0 ||
      launch.token_count > 4096 || launch.vector_dimension != 512 ||
      launch.ring_rows != 128 || launch.maximum_position_count == 0 ||
      launch.maximum_position_count > 1048576 ||
      launch.source_bf16 == launch.recent_ring_bf16) {
    return Status::InvalidArgument(
        "DeepSeek DSpark recent store launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
