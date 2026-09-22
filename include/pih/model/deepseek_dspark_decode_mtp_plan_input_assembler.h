#pragma once

#include <array>
#include <span>
#include <vector>

#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"
#include "pih/model/deepseek_dspark_device_resources.h"

namespace pih {

struct DeepSeekDsparkDecodeMtpPlanInput final {
  const DeepSeekDsparkWeightBindings* weights = nullptr;
  const DeepSeekDsparkResidentExpertBindings* resident_experts = nullptr;
  const DeepSeekFixedStateLayout* state_layout = nullptr;
  DeepSeekDsparkDeviceView device;
  DeepSeekAttentionProjectionDeviceView attention_workspace;
  DeepSeekMhcDeviceView mhc_workspace;
  std::uintptr_t rope_frequencies_f32 = 0;
  std::uintptr_t router_scores_f32 = 0;
  std::span<float> router_host_scores;
  std::span<float> router_host_bias;
  DeepSeekExpertComputeArena expert_arena;
  std::uintptr_t expert_accumulator_f32 = 0;
  DeepSeekDsparkExpertKernelDriver* expert_kernel = nullptr;
  std::array<DeepSeekDsparkMtpStageOperation*, kDeepSeekDsparkStageCount>
      attention_operations{};
  std::array<DeepSeekDsparkMtpStageOperation*, kDeepSeekDsparkStageCount>
      moe_operations{};
  std::uintptr_t stream = 0;
  std::uintptr_t completion_event = 0;
  std::uint32_t current_position = 0;
  std::uint32_t position_table_count = 0;
};

// Publishes the immutable, transaction-free decode template. The rank plan
// compiler binds the exact preparing transaction, prepare epoch and tentative
// recent-state bank immediately before execution.
class DeepSeekDsparkDecodeMtpPlanInputAssembler final {
 public:
  static Result<std::vector<DeepSeekBoundDsparkMtpStageWork>> Assemble(
      const DeepSeekDsparkDecodeMtpPlanInput& input);
};

}  // namespace pih
