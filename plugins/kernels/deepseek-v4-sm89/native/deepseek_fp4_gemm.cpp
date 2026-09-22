#include "pih/backend/cuda/deepseek_fp4_gemm.h"

namespace pih {

Status validate_deepseek_fp4_gemm_launch(
    const DeepSeekFp4GemmLaunch& launch) {
  const auto known_shape =
      (launch.n == 2048 && launch.k == 4096) ||
      (launch.n == 4096 && launch.k == 2048);
  const auto elements = static_cast<std::uint64_t>(launch.m) * launch.n;
  const auto blocks = (elements + 255U) / 256U;
  if (launch.activation_e4m3 == 0 || launch.activation_scale_bits == 0 ||
      launch.packed_weight == 0 || launch.weight_scale_bits == 0 ||
      launch.output_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.m == 0 || !known_shape ||
      blocks > 0x7FFFFFFFULL) {
    return Status::InvalidArgument("DeepSeek FP4 GEMM launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
