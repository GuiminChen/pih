#pragma once

#include <cstdint>

#include "pih/backend/cuda/deepseek_mhc.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

enum class DeepSeekMhcBranchKind : std::uint8_t {
  kAttention,
  kFeedForward,
};

struct DeepSeekMhcBranchLaunch final {
  DeepSeekMhcBranchKind kind = DeepSeekMhcBranchKind::kAttention;
  std::uint32_t layer_id = 0;
  std::uintptr_t input_bf16 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  std::uint32_t hidden_size = 0;
};

struct DeepSeekMhcSequenceSubmission final {
  DeepSeekMhcBranchKind kind = DeepSeekMhcBranchKind::kAttention;
  std::uint32_t layer_id = 0;
  std::uintptr_t residual_bf16 = 0;
  std::uintptr_t fn_f32 = 0;
  std::uintptr_t scale_f32 = 0;
  std::uintptr_t base_f32 = 0;
  std::uintptr_t norm_weight_bf16 = 0;
  std::uintptr_t post_mix_f32 = 0;
  std::uintptr_t residual_mix_f32 = 0;
  std::uintptr_t layer_input_bf16 = 0;
  std::uintptr_t branch_output_bf16 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t device_error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t token_count = 0;
  // Optional D-Spark target-hidden capture.  A non-zero destination is valid
  // only for feed-forward outputs of target layers 40, 41 and 42.
  std::uintptr_t target_hidden_bf16 = 0;
  std::uint32_t target_stage_index = 0;
  float rms_epsilon = 0.0F;
  float pre_epsilon = 0.0F;
  float sinkhorn_epsilon = 0.0F;
  std::uint32_t sinkhorn_iterations = 0;
};

Status validate_deepseek_mhc_sequence_submission(
    const DeepSeekMhcSequenceSubmission& submission);

class DeepSeekMhcSequenceOperations {
 public:
  virtual ~DeepSeekMhcSequenceOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status pre(DeepSeekMhcPreLaunch launch) = 0;
  virtual Status branch(DeepSeekMhcBranchLaunch launch) = 0;
  virtual Status post(DeepSeekMhcPostLaunch launch) = 0;
  virtual Status target_hidden_tap(
      DeepSeekMhcTargetHiddenTapLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};

class DeepSeekMhcSequenceExecutor final {
 public:
  static Result<DeepSeekMhcSequenceExecutor> Create(
      DeepSeekMhcSequenceOperations& operations,
      std::uint32_t* host_error_flag);

  Status launch(const DeepSeekMhcSequenceSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);
  Status begin(const DeepSeekMhcSequenceSubmission& submission,
               DeepSeekAttentionSequenceTransaction& transaction);
  Status finish(DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekMhcSequenceOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  DeepSeekAttentionSequenceTransaction* active_transaction_ = nullptr;
  DeepSeekMhcBranchLaunch active_branch_;
  DeepSeekMhcPostLaunch active_post_;
  DeepSeekMhcTargetHiddenTapLaunch active_target_hidden_tap_;
  bool poisoned_ = false;
};

}  // namespace pih
