#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include "pih/model/qwen3_teacher_forced_pair_completion.h"

namespace pih {

enum class QwenTeacherForcedPairChunkExecutorState : std::uint8_t {
  kAwaitingBf16, kBf16Pending, kAwaitingInt4, kInt4Pending,
  kFinalized, kPoisoned,
};

class QwenTeacherForcedPairChunkExecutor final {
 public:
  static Result<QwenTeacherForcedPairChunkExecutor> Create(
      std::span<const QwenTeacherForcedChunk> expected_chunks);

  Status submit_bf16(
      const QwenTeacherForcedChunk& chunk,
      QwenTeacherForcedChunkTransaction transaction,
      TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& head_driver,
      CompletionEventDriver& event_driver);
  Status submit_int4(
      const QwenTeacherForcedChunk& chunk,
      QwenTeacherForcedChunkTransaction transaction,
      TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenInt4LmHeadExecutionDriver& head_driver,
      CompletionEventDriver& event_driver);
  Status poll(CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns);
  Result<std::array<QwenTeacherForcedCategoryAggregate, 6>> finalize();

  [[nodiscard]] QwenTeacherForcedPairChunkExecutorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t completed_pairs() const noexcept {
    return completion_.completed_pairs();
  }

 private:
  explicit QwenTeacherForcedPairChunkExecutor(
      QwenTeacherForcedPairCompletion completion)
      : completion_(std::move(completion)) {}
  Status poison(std::string message);

  QwenTeacherForcedPairCompletion completion_;
  std::optional<QwenTeacherForcedChunkTransaction> active_;
  std::optional<QwenTeacherForcedChunk> active_chunk_;
  QwenTeacherForcedPairChunkExecutorState state_ =
      QwenTeacherForcedPairChunkExecutorState::kAwaitingBf16;
};

}  // namespace pih
