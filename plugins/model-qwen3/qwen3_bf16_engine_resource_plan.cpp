#include "pih/model/qwen3_bf16_engine_resource_plan.h"

#include <algorithm>
#include <array>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_bf16_linear_shape.h"
#include "pih/model/qwen3_manifest.h"

namespace pih {

Result<QwenBf16EngineResourcePlan> QwenBf16EngineResourcePlan::Create(
    std::uint32_t maximum_step_tokens,
    std::uint32_t maximum_sequence_tokens, std::uint32_t slot_count,
    std::uint64_t linear_workspace_bytes,
    std::uint32_t maximum_batch_sequences) {
  if (maximum_step_tokens == 0 ||
      maximum_step_tokens > QwenBf16LinearShape::kMaximumTokensPerPlan ||
      maximum_sequence_tokens < maximum_step_tokens ||
      maximum_sequence_tokens > QwenKvBlockTable::kMaximumReservedTokens ||
      slot_count == 0 || slot_count > QwenKvSlotPool::kMaximumSlots ||
      maximum_batch_sequences == 0 ||
      maximum_batch_sequences > maximum_step_tokens ||
      maximum_batch_sequences > QwenBf16PackedResultLayout::kMaximumSamples) {
    return Status::InvalidArgument("Qwen engine resource bounds are invalid");
  }
  const std::uint64_t required_slots =
      (static_cast<std::uint64_t>(maximum_sequence_tokens) +
       QwenKvSlotPool::kTokensPerSlot - 1) /
      QwenKvSlotPool::kTokensPerSlot;
  if (slot_count < required_slots) {
    return Status::InvalidArgument(
        "Qwen engine slot count cannot cover the sequence bound");
  }
  auto execution = QwenBf16ExecutionArenaLayout::Create(
      maximum_step_tokens, maximum_batch_sequences);
  if (!execution.ok()) return execution.status();
  auto staging = QwenBf16StepStagingLayout::CreateBounded(
      maximum_step_tokens, required_slots);
  if (!staging.ok()) return staging.status();
  auto packed_staging = QwenBf16PackedStepStagingLayout::CreateBounded(
      maximum_step_tokens, maximum_batch_sequences, slot_count);
  if (!packed_staging.ok()) return packed_staging.status();
  auto packed_result =
      QwenBf16PackedResultLayout::Create(maximum_batch_sequences);
  if (!packed_result.ok()) return packed_result.status();
  const auto step_staging_bytes = maximum_batch_sequences == 1
      ? staging->total_bytes()
      : (std::max)(staging->total_bytes(), packed_staging->total_bytes());
  const auto pinned_result_bytes = (std::max)(
      QwenBf16StepResultLayout::kTotalBytes, packed_result->total_bytes());
  const auto sampled_token_bytes = (std::max)(
      std::uint64_t{8},
      packed_result->device_error().offset_bytes);
  auto kv_backing = checked_mul_u64(slot_count,
                                    QwenKvSlotPool::kSlotPayloadBytes);
  auto kv_metadata = checked_mul_u64(slot_count, sizeof(QwenKvSlotState));
  if (!kv_backing.ok()) return kv_backing.status();
  if (!kv_metadata.ok()) return kv_metadata.status();
  const std::array<std::uint64_t, 11> allocations{
      step_staging_bytes, execution->activation_arena_bytes(),
      execution->mlp().arena_bytes(), execution->rope_workspace_bytes(),
      execution->logit_workspace_bytes() * 2, sampled_token_bytes,
      sizeof(std::uint32_t), *kv_backing, *kv_metadata,
      linear_workspace_bytes, Qwen3Manifest::kOfficialSourcePayloadBytes};
  std::uint64_t total = 0;
  for (const auto bytes : allocations) {
    auto next = checked_add_u64(total, bytes);
    if (!next.ok()) return next.status();
    total = *next;
  }
  return QwenBf16EngineResourcePlan(
      maximum_step_tokens, maximum_sequence_tokens, slot_count,
      std::move(*execution), std::move(*staging), std::move(*packed_staging),
      std::move(*packed_result), maximum_batch_sequences, step_staging_bytes,
      pinned_result_bytes, sampled_token_bytes, *kv_backing, *kv_metadata,
      linear_workspace_bytes, Qwen3Manifest::kOfficialSourcePayloadBytes,
      total);
}

}  // namespace pih
