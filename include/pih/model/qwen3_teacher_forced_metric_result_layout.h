#pragma once

#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_execution_arena.h"
#include "pih/model/qwen3_teacher_forced_metric_oracle.h"

namespace pih {

class QwenTeacherForcedMetricResultLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static constexpr std::uint32_t kMaximumRows = 4096;
  static constexpr std::uint32_t kVocabularySize = 151936;

  static Result<QwenTeacherForcedMetricResultLayout> Create(
      std::uint32_t row_capacity);
  Status initialize(std::span<std::byte> backing) const;
  Result<QwenTeacherForcedMetricBatch> parse(
      std::span<const std::byte> backing, std::uint32_t rows,
      bool publication_authorized) const;

  [[nodiscard]] std::uint32_t row_capacity() const noexcept {
    return row_capacity_;
  }
  [[nodiscard]] QwenBf16ArenaSpan argmax_tokens() const noexcept {
    return argmax_tokens_;
  }
  [[nodiscard]] QwenBf16ArenaSpan target_nll() const noexcept {
    return target_nll_;
  }
  [[nodiscard]] QwenBf16ArenaSpan nonfinite_rows() const noexcept {
    return nonfinite_rows_;
  }
  [[nodiscard]] QwenBf16ArenaSpan device_error() const noexcept {
    return device_error_;
  }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return total_bytes_;
  }

 private:
  QwenTeacherForcedMetricResultLayout(
      std::uint32_t row_capacity, QwenBf16ArenaSpan argmax_tokens,
      QwenBf16ArenaSpan target_nll, QwenBf16ArenaSpan nonfinite_rows,
      QwenBf16ArenaSpan device_error, std::uint64_t total_bytes)
      : row_capacity_(row_capacity), argmax_tokens_(argmax_tokens),
        target_nll_(target_nll), nonfinite_rows_(nonfinite_rows),
        device_error_(device_error), total_bytes_(total_bytes) {}

  std::uint32_t row_capacity_ = 0;
  QwenBf16ArenaSpan argmax_tokens_{};
  QwenBf16ArenaSpan target_nll_{};
  QwenBf16ArenaSpan nonfinite_rows_{};
  QwenBf16ArenaSpan device_error_{};
  std::uint64_t total_bytes_ = 0;
};

}  // namespace pih
