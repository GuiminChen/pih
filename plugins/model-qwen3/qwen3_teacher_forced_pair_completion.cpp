#include "pih/model/qwen3_teacher_forced_pair_completion.h"

#include <utility>

namespace pih {

Result<QwenTeacherForcedPairCompletion>
QwenTeacherForcedPairCompletion::Create(
    std::span<const QwenTeacherForcedChunk> expected_chunks) {
  auto collector = QwenTeacherForcedPairRunCollector::Create(expected_chunks);
  if (!collector.ok()) return collector.status();
  return QwenTeacherForcedPairCompletion(std::move(*collector));
}

Status QwenTeacherForcedPairCompletion::poison(std::string message) {
  poisoned_ = true;
  return Status::InvalidArgument(std::move(message));
}

Status QwenTeacherForcedPairCompletion::collect(
    const QwenTeacherForcedChunk& chunk,
    QwenTeacherForcedChunkTransaction& transaction,
    CompletionEventDriver& event_driver) {
  if (poisoned_) {
    return Status::FailedPrecondition("Qwen paired completion is poisoned");
  }
  const auto expected_role = awaiting_bf16_
      ? QwenTeacherForcedChunkRole::kBf16
      : QwenTeacherForcedChunkRole::kInt4;
  if (transaction.submitted_role() != expected_role) {
    return poison("Qwen paired completion transaction role differs");
  }
  auto batch = transaction.poll(event_driver);
  if (!batch.ok()) {
    if (batch.status().code() != StatusCode::kUnavailable) poisoned_ = true;
    return batch.status();
  }
  auto status = transaction.release_completion();
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  status = awaiting_bf16_
      ? collector_.stage_bf16(chunk, std::move(*batch))
      : collector_.stage_int4(chunk, std::move(*batch));
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  if (awaiting_bf16_) {
    awaiting_bf16_ = false;
  } else {
    awaiting_bf16_ = true;
    ++completed_pairs_;
  }
  return Status::Ok();
}

Status QwenTeacherForcedPairCompletion::expire(
    QwenTeacherForcedChunkTransaction& transaction, std::uint64_t now_ns) {
  if (poisoned_) {
    return Status::FailedPrecondition("Qwen paired completion is poisoned");
  }
  const auto expected_role = awaiting_bf16_
      ? QwenTeacherForcedChunkRole::kBf16
      : QwenTeacherForcedChunkRole::kInt4;
  if (transaction.submitted_role() != expected_role) {
    return poison("Qwen paired completion expiry role differs");
  }
  auto status = transaction.expire(now_ns);
  if (status.code() != StatusCode::kUnavailable) poisoned_ = true;
  return status;
}

Result<std::array<QwenTeacherForcedCategoryAggregate, 6>>
QwenTeacherForcedPairCompletion::finalize() {
  if (poisoned_ || !awaiting_bf16_) {
    return Status::FailedPrecondition("Qwen paired completion is incomplete");
  }
  return collector_.finalize();
}

}  // namespace pih
