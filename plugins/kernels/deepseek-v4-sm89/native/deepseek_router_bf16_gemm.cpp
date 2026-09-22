#include "pih/backend/cuda/deepseek_router_bf16_gemm.h"

namespace pih {

Status validate_deepseek_router_bf16_gemm_launch(
    const DeepSeekRouterBf16GemmLaunch& value) {
  if (value.input_bf16 == 0 || value.weight_bf16 == 0 ||
      value.scores_f32 == 0 || value.error_flag_u32 == 0 ||
      value.stream == 0 || value.token_count == 0 ||
      value.token_count > 4096 || value.expert_count != 256 ||
      value.hidden_size != 4096 || value.scores_f32 == value.input_bf16 ||
      value.scores_f32 == value.weight_bf16) {
    return Status::InvalidArgument(
        "DeepSeek BF16 router GEMM launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
