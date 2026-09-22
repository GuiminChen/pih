#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_teacher_forced_chunk_plan.h"

namespace pih {

struct QwenTeacherForcedTarget final {
  std::uint32_t sequence_index;
  std::uint32_t token_row;
  std::uint32_t target_token_id;
};

class QwenTeacherForcedBatchPlan final {
 public:
  static constexpr std::uint32_t kMaximumExecutionTokens = 4096;
  static constexpr std::uint32_t kMaximumSequences = 32;

  static Result<QwenTeacherForcedBatchPlan> Create(
      const QwenTeacherForcedChunk& chunk,
      std::span<const std::uint32_t> sequence_token_counts,
      std::span<const QwenTeacherForcedTarget> targets);

  Status materialize_inputs(std::span<std::byte> pinned_sample_rows,
                            std::span<std::byte> pinned_targets) const;

  [[nodiscard]] std::uint32_t execution_tokens() const noexcept {
    return execution_tokens_;
  }
  [[nodiscard]] const QwenTeacherForcedChunk& chunk() const noexcept {
    return chunk_;
  }
  [[nodiscard]] std::uint32_t sequence_count() const noexcept {
    return sequence_count_;
  }
  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return static_cast<std::uint32_t>(sample_row_indices_.size());
  }
  [[nodiscard]] std::span<const std::uint32_t> sample_row_indices() const noexcept {
    return sample_row_indices_;
  }
  [[nodiscard]] std::span<const std::uint32_t> target_token_ids() const noexcept {
    return target_token_ids_;
  }

 private:
  QwenTeacherForcedBatchPlan(QwenTeacherForcedChunk chunk,
                             std::uint32_t execution_tokens,
                             std::uint32_t sequence_count,
                             std::vector<std::uint32_t> sample_row_indices,
                             std::vector<std::uint32_t> target_token_ids)
      : chunk_(chunk), execution_tokens_(execution_tokens),
        sequence_count_(sequence_count),
        sample_row_indices_(std::move(sample_row_indices)),
        target_token_ids_(std::move(target_token_ids)) {}

  QwenTeacherForcedChunk chunk_{};
  std::uint32_t execution_tokens_ = 0;
  std::uint32_t sequence_count_ = 0;
  std::vector<std::uint32_t> sample_row_indices_;
  std::vector<std::uint32_t> target_token_ids_;
};

}  // namespace pih
