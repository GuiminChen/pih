#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {
struct DeepSeekHeadRmsLaunch final {
  std::uintptr_t input_bf16 = 0, output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, head_count = 0, head_dimension = 0;
  float epsilon = 0.0F;
};
struct DeepSeekRotaryLaunch final {
  std::uintptr_t input_bf16 = 0, frequencies_f32 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, head_count = 0, head_dimension = 0;
  std::uint32_t rope_dimension = 0;
  bool inverse = false;
  std::uintptr_t positions_u32 = 0;
  std::uint32_t table_position_count = 0;
};
struct DeepSeekKvFp8SimulateLaunch final {
  std::uintptr_t kv_bf16 = 0, error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, vector_dimension = 0;
  std::uint32_t quantized_dimension = 0, group_size = 0;
};
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
struct DeepSeekGroupedBf16GemmLaunch final {
  std::uintptr_t input_bf16 = 0, weight_bf16 = 0, output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, group_count = 0;
  std::uint32_t output_per_group = 0, input_per_group = 0;
};
#endif
struct DeepSeekGroupedFp8GemmLaunch final {
  std::uintptr_t input_bf16 = 0, activation_e4m3 = 0;
  std::uintptr_t activation_scale_bits = 0, weight_e4m3 = 0;
  std::uintptr_t weight_scale_bits = 0, output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, group_count = 0;
  std::uint32_t output_per_group = 0, input_per_group = 0;
};
Status validate_deepseek_head_rms_launch(const DeepSeekHeadRmsLaunch& launch);
Status validate_deepseek_rotary_launch(const DeepSeekRotaryLaunch& launch);
Status validate_deepseek_kv_fp8_simulate_launch(
    const DeepSeekKvFp8SimulateLaunch& launch);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status validate_deepseek_grouped_bf16_gemm_launch(
    const DeepSeekGroupedBf16GemmLaunch& launch);
#endif
Status validate_deepseek_grouped_fp8_gemm_launch(
    const DeepSeekGroupedFp8GemmLaunch& launch);
Status launch_deepseek_head_rms(DeepSeekHeadRmsLaunch launch);
Status launch_deepseek_rotary(DeepSeekRotaryLaunch launch);
Status launch_deepseek_kv_fp8_simulate(DeepSeekKvFp8SimulateLaunch launch);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status launch_deepseek_grouped_bf16_gemm(
    DeepSeekGroupedBf16GemmLaunch launch);
#endif
Status launch_deepseek_grouped_fp8_gemm(
    DeepSeekGroupedFp8GemmLaunch launch);
}  // namespace pih
