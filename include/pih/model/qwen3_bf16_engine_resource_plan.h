#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_execution_arena.h"
#include "pih/model/qwen3_bf16_step_result_layout.h"
#include "pih/model/qwen3_bf16_step_staging_layout.h"
#include "pih/model/qwen3_bf16_packed_result_layout.h"
#include "pih/model/qwen3_bf16_packed_step_staging_layout.h"

namespace pih {

class QwenBf16EngineResourcePlan final {
 public:
  static Result<QwenBf16EngineResourcePlan> Create(
      std::uint32_t maximum_step_tokens,
      std::uint32_t maximum_sequence_tokens, std::uint32_t slot_count,
      std::uint64_t linear_workspace_bytes,
      std::uint32_t maximum_batch_sequences = 1);

  [[nodiscard]] std::uint32_t maximum_step_tokens() const noexcept {
    return maximum_step_tokens_;
  }
  [[nodiscard]] std::uint32_t maximum_sequence_tokens() const noexcept {
    return maximum_sequence_tokens_;
  }
  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return slot_count_;
  }
  [[nodiscard]] std::uint64_t step_staging_bytes() const noexcept {
    return step_staging_bytes_;
  }
  [[nodiscard]] std::uint64_t pinned_result_bytes() const noexcept {
    return pinned_result_bytes_;
  }
  [[nodiscard]] std::uint64_t activation_bytes() const noexcept {
    return execution_.activation_arena_bytes();
  }
  [[nodiscard]] std::uint64_t mlp_bytes() const noexcept {
    return execution_.mlp().arena_bytes();
  }
  [[nodiscard]] std::uint64_t rope_bytes() const noexcept {
    return execution_.rope_workspace_bytes();
  }
  [[nodiscard]] std::uint64_t logits_bytes() const noexcept {
    return execution_.logit_workspace_bytes() * 2;
  }
  [[nodiscard]] std::uint64_t sampler_workspace_bytes() const noexcept {
    return execution_.logit_workspace_bytes();
  }
  [[nodiscard]] std::uint64_t sampled_token_bytes() const noexcept {
    return sampled_token_bytes_;
  }
  [[nodiscard]] std::uint64_t sampled_result_bytes() const noexcept {
    return sampled_token_bytes_;
  }
  [[nodiscard]] std::uint32_t maximum_batch_sequences() const noexcept {
    return maximum_batch_sequences_;
  }
  [[nodiscard]] const QwenBf16PackedStepStagingLayout&
  packed_staging_layout() const noexcept { return packed_staging_; }
  [[nodiscard]] const QwenBf16PackedResultLayout& packed_result_layout()
      const noexcept { return packed_result_; }
  [[nodiscard]] std::uint64_t device_error_bytes() const noexcept { return 4; }
  [[nodiscard]] std::uint64_t kv_backing_bytes() const noexcept {
    return kv_backing_bytes_;
  }
  [[nodiscard]] std::uint64_t kv_metadata_bytes() const noexcept {
    return kv_metadata_bytes_;
  }
  [[nodiscard]] std::uint64_t linear_workspace_bytes() const noexcept {
    return linear_workspace_bytes_;
  }
  [[nodiscard]] std::uint64_t resident_weight_bytes() const noexcept {
    return resident_weight_bytes_;
  }
  [[nodiscard]] std::uint64_t total_device_bytes() const noexcept {
    return total_device_bytes_;
  }
  [[nodiscard]] const QwenBf16ExecutionArenaLayout& execution_layout()
      const noexcept {
    return execution_;
  }
  [[nodiscard]] const QwenBf16StepStagingLayout& staging_layout()
      const noexcept {
    return staging_;
  }

 private:
  QwenBf16EngineResourcePlan(
      std::uint32_t maximum_step_tokens,
      std::uint32_t maximum_sequence_tokens, std::uint32_t slot_count,
      QwenBf16ExecutionArenaLayout execution,
      QwenBf16StepStagingLayout staging,
      QwenBf16PackedStepStagingLayout packed_staging,
      QwenBf16PackedResultLayout packed_result,
      std::uint32_t maximum_batch_sequences,
      std::uint64_t step_staging_bytes,
      std::uint64_t pinned_result_bytes,
      std::uint64_t sampled_token_bytes, std::uint64_t kv_backing_bytes,
      std::uint64_t kv_metadata_bytes, std::uint64_t linear_workspace_bytes,
      std::uint64_t resident_weight_bytes, std::uint64_t total_device_bytes)
      : maximum_step_tokens_(maximum_step_tokens),
        maximum_sequence_tokens_(maximum_sequence_tokens), slot_count_(slot_count),
        maximum_batch_sequences_(maximum_batch_sequences),
        execution_(std::move(execution)), staging_(std::move(staging)),
        packed_staging_(std::move(packed_staging)),
        packed_result_(std::move(packed_result)),
        step_staging_bytes_(step_staging_bytes),
        pinned_result_bytes_(pinned_result_bytes),
        sampled_token_bytes_(sampled_token_bytes),
        kv_backing_bytes_(kv_backing_bytes),
        kv_metadata_bytes_(kv_metadata_bytes),
        linear_workspace_bytes_(linear_workspace_bytes),
        resident_weight_bytes_(resident_weight_bytes),
        total_device_bytes_(total_device_bytes) {}

  std::uint32_t maximum_step_tokens_;
  std::uint32_t maximum_sequence_tokens_;
  std::uint32_t slot_count_;
  std::uint32_t maximum_batch_sequences_;
  QwenBf16ExecutionArenaLayout execution_;
  QwenBf16StepStagingLayout staging_;
  QwenBf16PackedStepStagingLayout packed_staging_;
  QwenBf16PackedResultLayout packed_result_;
  std::uint64_t step_staging_bytes_;
  std::uint64_t pinned_result_bytes_;
  std::uint64_t sampled_token_bytes_;
  std::uint64_t kv_backing_bytes_;
  std::uint64_t kv_metadata_bytes_;
  std::uint64_t linear_workspace_bytes_;
  std::uint64_t resident_weight_bytes_;
  std::uint64_t total_device_bytes_;
};

}  // namespace pih
