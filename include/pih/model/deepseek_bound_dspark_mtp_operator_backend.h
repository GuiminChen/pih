#pragma once

#include <span>

#include "pih/model/deepseek_dspark_mtp_block_executor.h"
#include "pih/model/deepseek_dspark_resident_expert_bindings.h"
#include "pih/model/deepseek_dspark_weight_bindings.h"
#include "pih/model/deepseek_fixed_state_layout.h"
#include "pih/model/deepseek_attention_projection_device_resources.h"
#include "pih/model/deepseek_mhc_device_resources.h"
#include "pih/model/deepseek_expert_compute_arena.h"

namespace pih {

class DeepSeekDsparkExpertKernelDriver;

struct DeepSeekDsparkMtpStageResources final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  const DeepSeekDsparkCommonWeightBindings* weights = nullptr;
  const DeepSeekDsparkResidentExpertBindings* resident_experts = nullptr;
  const DeepSeekFixedStateLayout* state_layout = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  std::uint64_t prepare_epoch = 0;
  DeepSeekDsparkFixedStageDeviceView recent_state;
  std::uint64_t weight_generation = 0;
  DeepSeekAttentionProjectionDeviceView attention_workspace;
  DeepSeekMhcDeviceView mhc_workspace;
  std::uintptr_t stage_input_hc_bf16 = 0;
  // Attention completes its mHC post into this intermediate.  The following
  // MoE operation consumes it and writes stage_output_hc_bf16.  Keeping the
  // addresses distinct prevents an in-place mHC post from corrupting the
  // residual mixture needed by either half of the block.
  std::uintptr_t attention_output_hc_bf16 = 0;
  std::uintptr_t stage_output_hc_bf16 = 0;
  std::uintptr_t main_normalized_bf16 = 0;
  std::uintptr_t main_activation_e4m3 = 0;
  std::uintptr_t main_activation_scale_ue8m0 = 0;
  std::uintptr_t main_kv_bf16 = 0;
  std::uintptr_t main_positions_u32 = 0;
  std::uintptr_t draft_positions_u32 = 0;
  std::uintptr_t rope_frequencies_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uintptr_t completion_event = 0;
  std::uintptr_t router_scores_f32 = 0;
  std::span<float> router_host_scores;
  std::span<float> router_host_bias;
  DeepSeekExpertComputeArena expert_arena;
  std::uintptr_t expert_accumulator_f32 = 0;
  DeepSeekDsparkExpertKernelDriver* expert_kernel = nullptr;
  std::uint32_t current_position = 0;
  std::uint32_t position_table_count = 0;
};

struct DeepSeekDsparkPrefillAttentionWeightBindings final {
  std::uintptr_t wkv_fp8 = 0;
  std::uintptr_t wkv_scale_ue8m0 = 0;
  std::uintptr_t kv_norm_bf16 = 0;
  std::uint64_t generation = 0;
};

// Minimal state-initialization surface from the pinned reference.  It has no
// router, expert, query/output projection, draft-token or head resource.
struct DeepSeekDsparkMtpPrefillResources final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  DeepSeekDsparkPrefillAttentionWeightBindings weights;
  const DeepSeekFixedStateLayout* state_layout = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  std::uint64_t prepare_epoch = 0;
  DeepSeekDsparkFixedStageDeviceView recent_state;
  std::uintptr_t main_normalized_bf16 = 0;
  std::uintptr_t activation_e4m3 = 0;
  std::uintptr_t activation_scale_ue8m0 = 0;
  std::uintptr_t positions_u32 = 0;
  std::uintptr_t kv_scratch_bf16 = 0;
  std::uintptr_t rope_frequencies_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uintptr_t completion_event = 0;
  std::uint32_t token_count = 0;
  std::uint32_t position_table_count = 0;
  std::uint64_t weight_generation = 0;
};

class DeepSeekDsparkMtpStageOperation {
 public:
  virtual ~DeepSeekDsparkMtpStageOperation() = default;
  // Once launch is entered, cancel must be safe even if launch returns an
  // error.  A failed launch may have partially submitted device work, so the
  // backend always invokes cancel before poisoning itself.  cancel must make
  // that work safe to release and must be idempotent for that launch attempt.
  virtual Status launch(
      const DeepSeekPipelinePlanDescriptor& plan,
      const DeepSeekDsparkMtpStageResources& resources) = 0;
  virtual Status launch_prefill(
      const DeepSeekPipelinePlanDescriptor& plan,
      const DeepSeekDsparkMtpPrefillResources& resources) = 0;
  virtual Result<DeepSeekStageComputeStatus> poll() = 0;
  virtual Status cancel() = 0;
};

struct DeepSeekBoundDsparkMtpStageWork final {
  DeepSeekDsparkMtpStageResources resources;
  DeepSeekDsparkMtpPrefillResources prefill_resources;
  DeepSeekDsparkMtpStageOperation* attention = nullptr;
  DeepSeekDsparkMtpStageOperation* moe = nullptr;
};

class DeepSeekDsparkMtpStageWorkProvider {
 public:
  virtual ~DeepSeekDsparkMtpStageWorkProvider() = default;
  // The returned work, resources, layout, transaction, bindings and operations
  // must remain alive and immutable until the matching operator reaches
  // success, error, or cancellation.
  virtual Result<const DeepSeekBoundDsparkMtpStageWork*> resolve(
      DeepSeekDsparkStageId stage,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

class DeepSeekBoundDsparkMtpOperatorBackend final
    : public DeepSeekDsparkMtpOperatorBackend {
 public:
  static Result<DeepSeekBoundDsparkMtpOperatorBackend> Create(
      DeepSeekDsparkMtpStageWorkProvider& provider);

  Status launch(const DeepSeekDsparkMtpOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;
  Status cancel() override;

 private:
  enum class State : std::uint8_t { kIdle, kInflight, kPoisoned };
  Status poison(Status status, bool cancel_operation);

  DeepSeekDsparkMtpStageWorkProvider* provider_ = nullptr;
  const DeepSeekBoundDsparkMtpStageWork* work_ = nullptr;
  DeepSeekDsparkMtpStageOperation* operation_ = nullptr;
  State state_ = State::kIdle;
};

}  // namespace pih
