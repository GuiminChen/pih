#include "pih/backend/cuda/deepseek_mhc.h"

#include <cmath>

namespace pih {

Status validate_deepseek_mhc_pre_launch(
    const DeepSeekMhcPreLaunch& launch) {
  if (launch.residual_bf16 == 0 || launch.fn_f32 == 0 ||
      launch.scale_f32 == 0 || launch.base_f32 == 0 ||
      launch.norm_weight_bf16 == 0 ||
      launch.post_mix_f32 == 0 || launch.residual_mix_f32 == 0 ||
      launch.layer_input_bf16 == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0 || launch.token_count == 0 ||
      launch.token_count > 4096 || launch.hidden_size != 4096 ||
      launch.layer_input_bf16 == launch.residual_bf16 ||
      launch.post_mix_f32 == launch.residual_mix_f32 ||
      !std::isfinite(launch.rms_epsilon) ||
      !std::isfinite(launch.pre_epsilon) ||
      !std::isfinite(launch.sinkhorn_epsilon) ||
      launch.rms_epsilon <= 0.0F || launch.pre_epsilon < 0.0F ||
      launch.sinkhorn_epsilon < 0.0F || launch.post_multiplier != 2.0F ||
      launch.sinkhorn_iterations != 20) {
    return Status::InvalidArgument("DeepSeek mHC pre launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_mhc_post_launch(
    const DeepSeekMhcPostLaunch& launch) {
  if (launch.layer_output_bf16 == 0 || launch.residual_bf16 == 0 ||
      launch.post_mix_f32 == 0 || launch.residual_mix_f32 == 0 ||
      launch.output_bf16 == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0 || launch.token_count == 0 ||
      launch.token_count > 4096 || launch.hidden_size != 4096 ||
      launch.output_bf16 == launch.residual_bf16 ||
      launch.output_bf16 == launch.layer_output_bf16) {
    return Status::InvalidArgument("DeepSeek mHC post launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_mhc_target_hidden_tap_launch(
    const DeepSeekMhcTargetHiddenTapLaunch& launch) {
  if (launch.residual_hc_bf16 == 0 || launch.target_hidden_bf16 == 0 ||
      launch.error_flag_u32 == 0 || launch.stream == 0 ||
      launch.token_count == 0 || launch.token_count > 4096 ||
      launch.hidden_size != 4096 || launch.source_stream_count != 4 ||
      launch.target_stage_count != 3 ||
      launch.target_stage_index >= launch.target_stage_count ||
      launch.residual_hc_bf16 == launch.target_hidden_bf16) {
    return Status::InvalidArgument(
        "DeepSeek mHC target hidden tap launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
