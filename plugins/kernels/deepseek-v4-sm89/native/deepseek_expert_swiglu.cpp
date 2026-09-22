#include "pih/backend/cuda/deepseek_expert_swiglu.h"

namespace pih {

Status validate_deepseek_expert_swiglu_launch(
    const DeepSeekExpertSwiGluLaunch& launch) {
  if (launch.gate_bf16 == 0 || launch.up_bf16 == 0 ||
      launch.route_weights_f32 == 0 || launch.output_bf16 == 0 ||
      launch.error_flag == 0 || launch.stream == 0 ||
      launch.token_count == 0) {
    return Status::InvalidArgument(
        "DeepSeek expert SwiGLU launch resources are invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_shared_expert_swiglu_launch(
    const DeepSeekSharedExpertSwiGluLaunch& launch) {
  if (launch.gate_bf16 == 0 || launch.up_bf16 == 0 ||
      launch.output_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.token_count == 0 ||
      launch.token_count > 4096) {
    return Status::InvalidArgument(
        "DeepSeek shared expert SwiGLU launch resources are invalid");
  }
  return Status::Ok();
}

}  // namespace pih
