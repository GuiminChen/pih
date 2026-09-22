#include "pih/backend/cuda/deepseek_fp8_gemm.h"

namespace pih {

Status validate_deepseek_fp8_gemm_launch(const DeepSeekFp8GemmLaunch& launch) {
  const auto elements = static_cast<std::uint64_t>(launch.m) * launch.n;
  const auto blocks = (elements + 255U) / 256U;
  if (launch.activation_e4m3 == 0 || launch.activation_scale_bits == 0 ||
      launch.weight_e4m3 == 0 || launch.weight_scale_bits == 0 ||
      launch.output_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.m == 0 || launch.m > 4096 ||
      launch.n == 0 || launch.n > 129280 || launch.n % 128U != 0 ||
      launch.k == 0 || launch.k > 32768 || launch.k % 128U != 0 ||
      (launch.output_type != DeepSeekFp8GemmOutputType::kBf16 &&
       launch.output_type != DeepSeekFp8GemmOutputType::kFp32) ||
      blocks > 0x7FFFFFFFULL) {
    return Status::InvalidArgument("DeepSeek FP8 GEMM launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
