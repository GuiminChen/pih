#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekRmsNormLaunch final {
  static constexpr float kEpsilon = 0.000001F;
  std::uintptr_t input_bf16 = 0;
  std::uintptr_t weight_bf16 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0;
  std::uint32_t hidden_size = 0;
  float epsilon = kEpsilon;
};

Status validate_deepseek_rms_norm_launch(const DeepSeekRmsNormLaunch& launch);
Status launch_deepseek_rms_norm(DeepSeekRmsNormLaunch launch);

struct DeepSeekFusedResidualRmsNormLaunch final {
  static constexpr float kEpsilon = DeepSeekRmsNormLaunch::kEpsilon;
  std::uintptr_t input_bf16 = 0;
  std::uintptr_t residual_bf16 = 0;
  std::uintptr_t weight_bf16 = 0;
  std::uintptr_t residual_output_bf16 = 0;
  std::uintptr_t normalized_output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0;
  std::uint32_t hidden_size = 0;
  float epsilon = kEpsilon;
};

Status validate_deepseek_fused_residual_rms_norm_launch(
    const DeepSeekFusedResidualRmsNormLaunch& launch);
Status launch_deepseek_fused_residual_rms_norm(
    DeepSeekFusedResidualRmsNormLaunch launch);
// Hardware-test-only locked fixture. It executes the fused candidate and the
// two-launch control path and requires bit-identical BF16 outputs.
Status cuda_verify_deepseek_fused_residual_rms_norm_fixture();

}  // namespace pih
