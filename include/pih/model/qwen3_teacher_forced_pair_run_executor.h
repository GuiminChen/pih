#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_teacher_forced_pair_chunk_executor.h"

namespace pih {

class QwenTeacherForcedPairRunExecutor final {
 public:
  static Result<QwenTeacherForcedPairRunExecutor> Create(
      std::span<const QwenTeacherForcedChunk> expected_chunks);

  Result<QwenTeacherForcedChunk> next_chunk() const;
  Status submit_bf16(
      QwenTeacherForcedChunkTransaction transaction,
      TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& head_driver,
      CompletionEventDriver& event_driver);
  Status submit_int4(
      QwenTeacherForcedChunkTransaction transaction,
      TypedCopyDriver& copy_driver,
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenInt4LmHeadExecutionDriver& head_driver,
      CompletionEventDriver& event_driver);
  Status poll(CompletionEventDriver& event_driver);
  Status expire(std::uint64_t now_ns) { return executor_.expire(now_ns); }
  Result<std::array<QwenTeacherForcedCategoryAggregate, 6>> finalize();

  [[nodiscard]] std::size_t completed_chunks() const noexcept { return next_; }

 private:
  QwenTeacherForcedPairRunExecutor(
      std::vector<QwenTeacherForcedChunk> expected,
      QwenTeacherForcedPairChunkExecutor executor)
      : expected_(std::move(expected)), executor_(std::move(executor)) {}

  std::vector<QwenTeacherForcedChunk> expected_;
  QwenTeacherForcedPairChunkExecutor executor_;
  std::size_t next_ = 0;
};

}  // namespace pih
