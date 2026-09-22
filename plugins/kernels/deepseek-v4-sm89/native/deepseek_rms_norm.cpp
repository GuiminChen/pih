#include "pih/backend/cuda/deepseek_rms_norm.h"

namespace pih {

Status validate_deepseek_rms_norm_launch(const DeepSeekRmsNormLaunch& launch) {
  if (launch.input_bf16 == 0 || launch.weight_bf16 == 0 ||
      launch.output_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.rows == 0 ||
      (launch.hidden_size != 512 && launch.hidden_size != 1024 &&
       launch.hidden_size != 4096) ||
      launch.epsilon != DeepSeekRmsNormLaunch::kEpsilon) {
    return Status::InvalidArgument("DeepSeek RMSNorm launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_fused_residual_rms_norm_launch(
    const DeepSeekFusedResidualRmsNormLaunch& launch) {
  if (launch.input_bf16 == 0 || launch.residual_bf16 == 0 ||
      launch.weight_bf16 == 0 || launch.residual_output_bf16 == 0 ||
      launch.normalized_output_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.rows == 0 || launch.rows > 4096 ||
      launch.hidden_size != 4096 ||
      launch.epsilon != DeepSeekFusedResidualRmsNormLaunch::kEpsilon ||
      launch.residual_output_bf16 == launch.input_bf16 ||
      launch.residual_output_bf16 == launch.residual_bf16 ||
      launch.normalized_output_bf16 == launch.input_bf16 ||
      launch.normalized_output_bf16 == launch.residual_bf16 ||
      launch.normalized_output_bf16 == launch.residual_output_bf16) {
    return Status::InvalidArgument(
        "DeepSeek fused residual RMSNorm launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
