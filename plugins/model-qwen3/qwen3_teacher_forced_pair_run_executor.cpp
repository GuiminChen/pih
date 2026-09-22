#include "pih/model/qwen3_teacher_forced_pair_run_executor.h"

#include <utility>

namespace pih {

Result<QwenTeacherForcedPairRunExecutor>
QwenTeacherForcedPairRunExecutor::Create(
    std::span<const QwenTeacherForcedChunk> expected_chunks) {
  auto executor = QwenTeacherForcedPairChunkExecutor::Create(expected_chunks);
  if (!executor.ok()) return executor.status();
  return QwenTeacherForcedPairRunExecutor(
      std::vector<QwenTeacherForcedChunk>(expected_chunks.begin(),
                                          expected_chunks.end()),
      std::move(*executor));
}

Result<QwenTeacherForcedChunk>
QwenTeacherForcedPairRunExecutor::next_chunk() const {
  if (next_ >= expected_.size())
    return Status::FailedPrecondition("Qwen paired run plan is exhausted");
  return expected_[next_];
}

Status QwenTeacherForcedPairRunExecutor::submit_bf16(
    QwenTeacherForcedChunkTransaction transaction,
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  auto chunk = next_chunk();
  if (!chunk.ok()) return chunk.status();
  return executor_.submit_bf16(
      *chunk, std::move(transaction), copy_driver, clear_driver,
      kernel_driver, head_driver, event_driver);
}

Status QwenTeacherForcedPairRunExecutor::submit_int4(
    QwenTeacherForcedChunkTransaction transaction,
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  auto chunk = next_chunk();
  if (!chunk.ok()) return chunk.status();
  return executor_.submit_int4(
      *chunk, std::move(transaction), copy_driver, clear_driver,
      kernel_driver, head_driver, event_driver);
}

Status QwenTeacherForcedPairRunExecutor::poll(
    CompletionEventDriver& event_driver) {
  const auto previous = executor_.state();
  auto status = executor_.poll(event_driver);
  if (!status.ok()) return status;
  if (previous == QwenTeacherForcedPairChunkExecutorState::kInt4Pending)
    ++next_;
  return Status::Ok();
}

Result<std::array<QwenTeacherForcedCategoryAggregate, 6>>
QwenTeacherForcedPairRunExecutor::finalize() {
  if (next_ != expected_.size())
    return Status::FailedPrecondition("Qwen paired run plan is incomplete");
  return executor_.finalize();
}

}  // namespace pih
