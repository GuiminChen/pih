#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class QwenTeacherForcedCategory : std::uint8_t {
  kZh = 0, kEn, kCode, kTool, kThinking, kNonThinking,
};

struct QwenTeacherForcedChunk final {
  QwenTeacherForcedCategory category;
  std::uint64_t category_row_begin;
  std::uint32_t rows;
  std::uint64_t logits_bytes;
};

class QwenTeacherForcedChunkPlan final {
 public:
  static constexpr std::uint32_t kCategoryCount = 6;
  static constexpr std::uint64_t kMinimumRowsPerCategory = 100000;
  static constexpr std::uint64_t kMaximumRowsPerCategory = 10000000;
  static constexpr std::uint32_t kMaximumRowsPerChunk = 32;
  static constexpr std::uint64_t kVocabularySize = 151936;

  static Result<QwenTeacherForcedChunkPlan> Create(
      std::span<const std::uint64_t, kCategoryCount> category_rows,
      std::uint32_t maximum_rows_per_chunk = kMaximumRowsPerChunk);

  [[nodiscard]] std::span<const QwenTeacherForcedChunk> chunks() const noexcept {
    return chunks_;
  }
  [[nodiscard]] const std::array<std::uint64_t, kCategoryCount>&
  category_rows() const noexcept { return category_rows_; }
  [[nodiscard]] std::uint64_t total_rows() const noexcept { return total_rows_; }
  [[nodiscard]] std::uint32_t maximum_rows_per_chunk() const noexcept {
    return maximum_rows_per_chunk_;
  }

 private:
  QwenTeacherForcedChunkPlan(
      std::array<std::uint64_t, kCategoryCount> category_rows,
      std::vector<QwenTeacherForcedChunk> chunks, std::uint64_t total_rows,
      std::uint32_t maximum_rows_per_chunk)
      : category_rows_(category_rows), chunks_(std::move(chunks)),
        total_rows_(total_rows), maximum_rows_per_chunk_(maximum_rows_per_chunk) {}
  std::array<std::uint64_t, kCategoryCount> category_rows_{};
  std::vector<QwenTeacherForcedChunk> chunks_;
  std::uint64_t total_rows_ = 0;
  std::uint32_t maximum_rows_per_chunk_ = 0;
};

}  // namespace pih
