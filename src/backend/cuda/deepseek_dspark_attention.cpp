#include "pih/backend/cuda/deepseek_dspark_attention.h"

namespace pih {

Status validate_deepseek_dspark_attention_launch(
    const DeepSeekDsparkAttentionLaunch& launch) {
  if (launch.query_bf16 == 0 || launch.recent_kv_bf16 == 0 ||
      launch.draft_kv_bf16 == 0 || launch.attention_sink_f32 == 0 ||
      launch.output_bf16 == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0 ||
      launch.query_count != DeepSeekDsparkAttentionLaunch::kBlockSize ||
      launch.recent_count == 0 ||
      launch.recent_count > DeepSeekDsparkAttentionLaunch::kWindowSize) {
    return Status::InvalidArgument(
        "DeepSeek DSpark attention launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_dspark_position_launch(
    const DeepSeekDsparkPositionLaunch& launch) {
  constexpr auto block = DeepSeekDsparkAttentionLaunch::kBlockSize;
  if (launch.draft_positions_u32 == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0 || launch.table_position_count == 0 ||
      launch.current_position >= launch.table_position_count ||
      launch.current_position > UINT32_MAX - block ||
      launch.current_position + block >= launch.table_position_count) {
    return Status::InvalidArgument(
        "DeepSeek DSpark position launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
