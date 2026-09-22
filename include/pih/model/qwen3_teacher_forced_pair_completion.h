#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "pih/model/qwen3_teacher_forced_chunk_transaction.h"
#include "pih/model/qwen3_teacher_forced_pair_accumulator.h"

namespace pih {

class QwenTeacherForcedPairCompletion final {
 public:
  static Result<QwenTeacherForcedPairCompletion> Create(
      std::span<const QwenTeacherForcedChunk> expected_chunks);

  Status collect(const QwenTeacherForcedChunk& chunk,
                 QwenTeacherForcedChunkTransaction& transaction,
                 CompletionEventDriver& event_driver);
  Status expire(QwenTeacherForcedChunkTransaction& transaction,
                std::uint64_t now_ns);
  Result<std::array<QwenTeacherForcedCategoryAggregate, 6>> finalize();

  [[nodiscard]] std::uint64_t completed_pairs() const noexcept {
    return completed_pairs_;
  }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  explicit QwenTeacherForcedPairCompletion(
      QwenTeacherForcedPairRunCollector collector)
      : collector_(std::move(collector)) {}
  Status poison(std::string message);

  QwenTeacherForcedPairRunCollector collector_;
  bool awaiting_bf16_ = true;
  bool poisoned_ = false;
  std::uint64_t completed_pairs_ = 0;
};

}  // namespace pih
