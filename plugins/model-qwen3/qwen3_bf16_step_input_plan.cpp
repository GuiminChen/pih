#include "pih/model/qwen3_bf16_step_input_plan.h"

namespace pih {

Result<QwenBf16StepInputPlan> QwenBf16StepInputPlan::Create(
    std::span<const std::int64_t> tokens, std::uint64_t first_position,
    const QwenKvBlockTable& block_table,
    const QwenKvAppendPlan& append_plan) {
  const auto& descriptor = block_table.descriptor();
  if (tokens.empty() || tokens.size() > kMaximumStepTokens ||
      first_position != descriptor.committed_tokens ||
      first_position > QwenKvBlockTable::kMaximumReservedTokens ||
      tokens.size() > QwenKvBlockTable::kMaximumReservedTokens -
                          first_position ||
      append_plan.owner_sequence_index != descriptor.owner_sequence_index ||
      append_plan.expected_sequence_generation !=
          descriptor.sequence_generation ||
      append_plan.expected_block_table_generation !=
          descriptor.block_table_generation ||
      append_plan.previous_committed_tokens != descriptor.committed_tokens ||
      append_plan.target_committed_tokens != first_position + tokens.size() ||
      append_plan.target_committed_tokens > descriptor.reserved_tokens) {
    return Status::InvalidArgument(
        "Qwen step input does not match the append transaction");
  }
  const auto reserved = block_table.reserved_handles();
  const std::uint32_t required_handles =
      (append_plan.target_committed_tokens +
       QwenKvSlotPool::kTokensPerSlot - 1U) /
      QwenKvSlotPool::kTokensPerSlot;
  if (append_plan.visible_handle_count_after_commit != required_handles ||
      required_handles > reserved.size()) {
    return Status::InvalidArgument(
        "Qwen step input handle coverage is invalid");
  }

  QwenBf16StepInputPlan result;
  result.tokens_.assign(tokens.begin(), tokens.end());
  result.positions_.reserve(tokens.size());
  result.append_handles_.reserve(tokens.size());
  result.token_offsets_.reserve(tokens.size());
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (tokens[index] < 0 || tokens[index] >= 151936) {
      return Status::InvalidArgument(
          "Qwen step input contains a token outside the vocabulary");
    }
    const std::uint64_t position = first_position + index;
    result.positions_.push_back(static_cast<std::int64_t>(position));
    result.append_handles_.push_back(
        reserved[position / QwenKvSlotPool::kTokensPerSlot]);
    result.token_offsets_.push_back(static_cast<std::uint16_t>(
        position % QwenKvSlotPool::kTokensPerSlot));
  }
  result.visible_handles_.assign(reserved.begin(),
                                 reserved.begin() + required_handles);
  result.key_token_count_ = append_plan.target_committed_tokens;
  return result;
}

}  // namespace pih
