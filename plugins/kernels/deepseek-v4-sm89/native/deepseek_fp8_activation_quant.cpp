#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"

namespace pih {

Status validate_deepseek_fp8_activation_quant_launch(
    const DeepSeekFp8ActivationQuantLaunch& launch) {
  const auto groups_per_token =
      launch.logical_k / DeepSeekFp8ActivationQuantLaunch::kGroupSize;
  const auto grid_x = static_cast<std::uint64_t>(launch.token_count) *
                      groups_per_token;
  if (launch.input_bf16 == 0 || launch.output_e4m3 == 0 ||
      launch.scale_bits == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.token_count == 0 ||
      (launch.logical_k != 1024 && launch.logical_k != 2048 &&
       launch.logical_k != 4096 && launch.logical_k != 8192 &&
       launch.logical_k != 12288) ||
      grid_x > 0x7FFFFFFFULL) {
    return Status::InvalidArgument(
        "DeepSeek FP8 activation quant launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
