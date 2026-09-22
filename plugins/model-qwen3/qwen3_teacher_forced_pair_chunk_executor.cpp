#include "pih/model/qwen3_teacher_forced_pair_chunk_executor.h"

#include <utility>

namespace pih {

Result<QwenTeacherForcedPairChunkExecutor>
QwenTeacherForcedPairChunkExecutor::Create(
    std::span<const QwenTeacherForcedChunk> expected_chunks) {
  auto completion = QwenTeacherForcedPairCompletion::Create(expected_chunks);
  if (!completion.ok()) return completion.status();
  return QwenTeacherForcedPairChunkExecutor(std::move(*completion));
}

Status QwenTeacherForcedPairChunkExecutor::poison(std::string message) {
  state_ = QwenTeacherForcedPairChunkExecutorState::kPoisoned;
  active_.reset();
  active_chunk_.reset();
  return Status::InvalidArgument(std::move(message));
}

Status QwenTeacherForcedPairChunkExecutor::submit_bf16(
    const QwenTeacherForcedChunk& chunk,
    QwenTeacherForcedChunkTransaction transaction,
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  if (state_ != QwenTeacherForcedPairChunkExecutorState::kAwaitingBf16)
    return poison("Qwen paired chunk executor BF16 order is invalid");
  auto status = transaction.submit_bf16(
      copy_driver, clear_driver, kernel_driver, head_driver, event_driver);
  if (!status.ok()) {
    state_ = QwenTeacherForcedPairChunkExecutorState::kPoisoned;
    return status;
  }
  active_chunk_ = chunk;
  active_.emplace(std::move(transaction));
  state_ = QwenTeacherForcedPairChunkExecutorState::kBf16Pending;
  return Status::Ok();
}

Status QwenTeacherForcedPairChunkExecutor::submit_int4(
    const QwenTeacherForcedChunk& chunk,
    QwenTeacherForcedChunkTransaction transaction,
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  if (state_ != QwenTeacherForcedPairChunkExecutorState::kAwaitingInt4)
    return poison("Qwen paired chunk executor INT4 order is invalid");
  if (!active_chunk_.has_value() ||
      active_chunk_->category != chunk.category ||
      active_chunk_->category_row_begin != chunk.category_row_begin ||
      active_chunk_->rows != chunk.rows ||
      active_chunk_->logits_bytes != chunk.logits_bytes)
    return poison("Qwen paired chunk executor chunk identity differs");
  auto status = transaction.submit_int4(
      copy_driver, clear_driver, kernel_driver, head_driver, event_driver);
  if (!status.ok()) {
    state_ = QwenTeacherForcedPairChunkExecutorState::kPoisoned;
    return status;
  }
  active_.emplace(std::move(transaction));
  state_ = QwenTeacherForcedPairChunkExecutorState::kInt4Pending;
  return Status::Ok();
}

Status QwenTeacherForcedPairChunkExecutor::poll(
    CompletionEventDriver& event_driver) {
  if ((state_ != QwenTeacherForcedPairChunkExecutorState::kBf16Pending &&
       state_ != QwenTeacherForcedPairChunkExecutorState::kInt4Pending) ||
      !active_.has_value() || !active_chunk_.has_value())
    return poison("Qwen paired chunk executor has no pending transaction");
  auto status = completion_.collect(*active_chunk_, *active_, event_driver);
  if (!status.ok()) {
    if (status.code() != StatusCode::kUnavailable)
      state_ = QwenTeacherForcedPairChunkExecutorState::kPoisoned;
    return status;
  }
  active_.reset();
  if (state_ == QwenTeacherForcedPairChunkExecutorState::kBf16Pending) {
    state_ = QwenTeacherForcedPairChunkExecutorState::kAwaitingInt4;
  } else {
    active_chunk_.reset();
    state_ = QwenTeacherForcedPairChunkExecutorState::kAwaitingBf16;
  }
  return Status::Ok();
}

Status QwenTeacherForcedPairChunkExecutor::expire(std::uint64_t now_ns) {
  if (!active_.has_value())
    return poison("Qwen paired chunk executor has no expirable transaction");
  auto status = completion_.expire(*active_, now_ns);
  if (status.code() != StatusCode::kUnavailable)
    state_ = QwenTeacherForcedPairChunkExecutorState::kPoisoned;
  return status;
}

Result<std::array<QwenTeacherForcedCategoryAggregate, 6>>
QwenTeacherForcedPairChunkExecutor::finalize() {
  if (state_ != QwenTeacherForcedPairChunkExecutorState::kAwaitingBf16)
    return Status::FailedPrecondition("Qwen paired chunk executor is incomplete");
  auto result = completion_.finalize();
  if (!result.ok()) return result.status();
  state_ = QwenTeacherForcedPairChunkExecutorState::kFinalized;
  return result;
}

}  // namespace pih
