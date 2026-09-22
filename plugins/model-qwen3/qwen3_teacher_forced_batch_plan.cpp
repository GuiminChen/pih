#include "pih/model/qwen3_teacher_forced_batch_plan.h"

#include <cstring>
#include <vector>

namespace pih {

Result<QwenTeacherForcedBatchPlan> QwenTeacherForcedBatchPlan::Create(
    const QwenTeacherForcedChunk& chunk,
    std::span<const std::uint32_t> sequence_token_counts,
    std::span<const QwenTeacherForcedTarget> targets) {
  if (chunk.rows == 0 ||
      chunk.rows > QwenTeacherForcedChunkPlan::kMaximumRowsPerChunk ||
      targets.size() != chunk.rows || sequence_token_counts.empty() ||
      sequence_token_counts.size() > kMaximumSequences ||
      chunk.logits_bytes != static_cast<std::uint64_t>(chunk.rows) *
                                QwenTeacherForcedChunkPlan::kVocabularySize *
                                sizeof(float)) {
    return Status::InvalidArgument(
        "Qwen teacher-forced batch identity is invalid");
  }

  std::vector<std::uint32_t> row_offsets(sequence_token_counts.size());
  std::uint32_t execution_tokens = 0;
  for (std::size_t index = 0; index < sequence_token_counts.size(); ++index) {
    const auto count = sequence_token_counts[index];
    if (count == 0 || count > kMaximumExecutionTokens ||
        execution_tokens > kMaximumExecutionTokens - count) {
      return Status::ResourceExhausted(
          "Qwen teacher-forced packed token capacity is exceeded");
    }
    row_offsets[index] = execution_tokens;
    execution_tokens += count;
  }

  std::vector<std::uint32_t> sample_rows;
  std::vector<std::uint32_t> target_ids;
  sample_rows.reserve(targets.size());
  target_ids.reserve(targets.size());
  for (const auto& target : targets) {
    if (target.sequence_index >= sequence_token_counts.size() ||
        target.token_row >= sequence_token_counts[target.sequence_index] ||
        target.target_token_id >=
            QwenTeacherForcedChunkPlan::kVocabularySize) {
      return Status::InvalidArgument(
          "Qwen teacher-forced causal target is invalid");
    }
    sample_rows.push_back(row_offsets[target.sequence_index] + target.token_row);
    target_ids.push_back(target.target_token_id);
  }
  return QwenTeacherForcedBatchPlan(
      chunk, execution_tokens,
      static_cast<std::uint32_t>(sequence_token_counts.size()),
      std::move(sample_rows), std::move(target_ids));
}

Status QwenTeacherForcedBatchPlan::materialize_inputs(
    std::span<std::byte> pinned_sample_rows,
    std::span<std::byte> pinned_targets) const {
  const auto bytes = sample_row_indices_.size() * sizeof(std::uint32_t);
  if (bytes == 0 || pinned_sample_rows.size() < bytes ||
      pinned_targets.size() < bytes ||
      target_token_ids_.size() != sample_row_indices_.size()) {
    return Status::InvalidArgument(
        "Qwen teacher-forced pinned input capacity is invalid");
  }
  std::memcpy(pinned_sample_rows.data(), sample_row_indices_.data(), bytes);
  std::memcpy(pinned_targets.data(), target_token_ids_.data(), bytes);
  return Status::Ok();
}

}  // namespace pih
