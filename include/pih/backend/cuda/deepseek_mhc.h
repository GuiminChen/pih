#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekMhcPreLaunch final {
  std::uintptr_t residual_bf16 = 0;
  std::uintptr_t fn_f32 = 0;
  std::uintptr_t scale_f32 = 0;
  std::uintptr_t base_f32 = 0;
  std::uintptr_t norm_weight_bf16 = 0;
  std::uintptr_t post_mix_f32 = 0;
  std::uintptr_t residual_mix_f32 = 0;
  std::uintptr_t layer_input_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t hidden_size = 0;
  float rms_epsilon = 0.0F;
  float pre_epsilon = 0.0F;
  float sinkhorn_epsilon = 0.0F;
  float post_multiplier = 0.0F;
  std::uint32_t sinkhorn_iterations = 0;
};

struct DeepSeekMhcPostLaunch final {
  std::uintptr_t layer_output_bf16 = 0;
  std::uintptr_t residual_bf16 = 0;
  std::uintptr_t post_mix_f32 = 0;
  std::uintptr_t residual_mix_f32 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t hidden_size = 0;
};

// Captures the model hidden state consumed by the three-stage D-Spark path.
// The source is the four-stream mHC residual.  Each target layer contributes
// the arithmetic mean of those streams to one contiguous [token, 3, hidden]
// destination slice.
struct DeepSeekMhcTargetHiddenTapLaunch final {
  std::uintptr_t residual_hc_bf16 = 0;
  std::uintptr_t target_hidden_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t source_stream_count = 0;
  std::uint32_t target_stage_count = 0;
  std::uint32_t target_stage_index = 0;
};

Status validate_deepseek_mhc_pre_launch(const DeepSeekMhcPreLaunch& launch);
Status validate_deepseek_mhc_post_launch(const DeepSeekMhcPostLaunch& launch);
Status validate_deepseek_mhc_target_hidden_tap_launch(
    const DeepSeekMhcTargetHiddenTapLaunch& launch);
Status launch_deepseek_mhc_pre(DeepSeekMhcPreLaunch launch);
Status launch_deepseek_mhc_post(DeepSeekMhcPostLaunch launch);
Status launch_deepseek_mhc_target_hidden_tap(
    DeepSeekMhcTargetHiddenTapLaunch launch);

}  // namespace pih
