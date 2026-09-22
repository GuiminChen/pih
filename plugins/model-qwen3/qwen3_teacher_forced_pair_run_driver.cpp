#include "pih/model/qwen3_teacher_forced_pair_run_driver.h"

#include "pih/core/checked_math.h"

#include <utility>

namespace pih {
namespace {

bool same_chunk(const QwenTeacherForcedChunk& left,
                const QwenTeacherForcedChunk& right) {
  return left.category == right.category &&
         left.category_row_begin == right.category_row_begin &&
         left.rows == right.rows && left.logits_bytes == right.logits_bytes;
}

Status validate_run_identity_capacity(
    std::size_t chunk_count,
    const QwenTeacherForcedPairRunIdentity& identity) {
  auto reservations = checked_mul_u64(
      static_cast<std::uint64_t>(chunk_count), UINT64_C(2));
  if (!reservations.ok()) return reservations.status();
  auto plan_advance = checked_mul_u64(*reservations, UINT64_C(4));
  if (!plan_advance.ok()) return plan_advance.status();
  auto final_plan = checked_add_u64(identity.first_plan_id, *plan_advance);
  if (!final_plan.ok()) return final_plan.status();
  auto final_event = checked_add_u64(
      identity.first_event_generation, *reservations);
  if (!final_event.ok()) return final_event.status();
  auto final_frontier = checked_add_u64(
      identity.first_completion_frontier, *reservations);
  if (!final_frontier.ok()) return final_frontier.status();
  return Status::Ok();
}

}  // namespace

Result<QwenTeacherForcedPairRunDriver>
QwenTeacherForcedPairRunDriver::Create(
    std::span<const QwenTeacherForcedChunk> expected_chunks,
    QwenTeacherForcedPairRunIdentity identity) {
  if (identity.engine_epoch == 0 || identity.event == 0 ||
      identity.context_identity == 0 || identity.first_plan_id == 0 ||
      identity.first_event_generation == 0 ||
      identity.first_completion_frontier == 0)
    return Status::InvalidArgument("Qwen paired run driver identity is invalid");
  auto executor = QwenTeacherForcedPairRunExecutor::Create(expected_chunks);
  if (!executor.ok()) return executor.status();
  auto capacity = validate_run_identity_capacity(expected_chunks.size(),
                                                 identity);
  if (!capacity.ok()) return capacity;
  return QwenTeacherForcedPairRunDriver(std::move(*executor), identity);
}

Status QwenTeacherForcedPairRunDriver::validate_batch(
    const QwenTeacherForcedBatchPlan& batch) const {
  if (poisoned_)
    return Status::FailedPrecondition("Qwen paired run driver is poisoned");
  auto expected = executor_.next_chunk();
  if (!expected.ok()) return expected.status();
  if (!same_chunk(*expected, batch.chunk()))
    return Status::InvalidArgument("Qwen paired run batch chunk differs");
  return Status::Ok();
}

Status QwenTeacherForcedPairRunDriver::validate_factory(
    const QwenTeacherForcedTransactionFactory& factory) const {
  if (factory_capability_.has_value() &&
      *factory_capability_ != factory.capability())
    return Status::InvalidArgument("Qwen paired run factory identity differs");
  return Status::Ok();
}

Result<QwenTeacherForcedChunkTransaction>
QwenTeacherForcedPairRunDriver::build(
    const QwenTeacherForcedBatchPlan& batch,
    QwenTeacherForcedTransactionFactory& factory,
    std::uint64_t submit_ns, std::uint64_t deadline_ns,
    QwenBf16StepHealthProvider& health_provider,
    QwenTeacherForcedTransactionIdentityReservation* reservation,
    std::uint64_t* next_frontier) {
  auto status = validate_batch(batch);
  if (!status.ok()) return status;
  status = validate_factory(factory);
  if (!status.ok()) return status;
  auto reserved = reserve_qwen_teacher_forced_transaction_identity(
      next_plan_id_, next_event_generation_);
  if (!reserved.ok()) return reserved.status();
  auto following_frontier = checked_add_u64(next_completion_frontier_, 1);
  if (!following_frontier.ok()) return following_frontier.status();
  auto completion = make_qwen_teacher_forced_transaction_completion(
      *reserved, identity_.event, identity_.context_identity,
      identity_.engine_epoch, identity_.rank, next_completion_frontier_,
      submit_ns, deadline_ns);
  if (!completion.ok()) return completion.status();
  auto transaction = factory.build_reserved(
      batch, *reserved, std::move(*completion), health_provider);
  if (!transaction.ok()) return transaction.status();
  *reservation = *reserved;
  *next_frontier = *following_frontier;
  return transaction;
}

void QwenTeacherForcedPairRunDriver::commit(
    const QwenTeacherForcedTransactionIdentityReservation& value,
    std::uint64_t next_frontier) noexcept {
  next_plan_id_ = value.next_plan_id;
  next_event_generation_ = value.next_event_generation;
  next_completion_frontier_ = next_frontier;
}

Status QwenTeacherForcedPairRunDriver::submit_bf16(
    const QwenTeacherForcedBatchPlan& batch,
    QwenTeacherForcedTransactionFactory& factory,
    std::uint64_t submit_ns, std::uint64_t deadline_ns,
    QwenBf16StepHealthProvider& health_provider,
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  if (poisoned_)
    return Status::FailedPrecondition("Qwen paired run driver is poisoned");
  if (phase_ != QwenTeacherForcedPairRunDriverPhase::kAwaitingBf16)
    return Status::FailedPrecondition(
        "Qwen paired run driver is not awaiting BF16");
  QwenTeacherForcedTransactionIdentityReservation reservation{};
  std::uint64_t next_frontier = 0;
  auto transaction = build(batch, factory, submit_ns, deadline_ns,
                           health_provider, &reservation, &next_frontier);
  if (!transaction.ok()) return transaction.status();
  auto status = executor_.submit_bf16(
      std::move(*transaction), copy_driver, clear_driver, kernel_driver,
      head_driver, event_driver);
  if (!status.ok()) {
    poisoned_ = true;
    phase_ = QwenTeacherForcedPairRunDriverPhase::kPoisoned;
    return status;
  }
  if (!factory_capability_.has_value())
    factory_capability_ = factory.capability();
  commit(reservation, next_frontier);
  phase_ = QwenTeacherForcedPairRunDriverPhase::kBf16Pending;
  return Status::Ok();
}

Status QwenTeacherForcedPairRunDriver::submit_int4(
    const QwenTeacherForcedBatchPlan& batch,
    QwenTeacherForcedTransactionFactory& factory,
    std::uint64_t submit_ns, std::uint64_t deadline_ns,
    QwenBf16StepHealthProvider& health_provider,
    TypedCopyDriver& copy_driver,
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& head_driver,
    CompletionEventDriver& event_driver) {
  if (poisoned_)
    return Status::FailedPrecondition("Qwen paired run driver is poisoned");
  if (phase_ != QwenTeacherForcedPairRunDriverPhase::kAwaitingInt4)
    return Status::FailedPrecondition(
        "Qwen paired run driver is not awaiting INT4");
  QwenTeacherForcedTransactionIdentityReservation reservation{};
  std::uint64_t next_frontier = 0;
  auto transaction = build(batch, factory, submit_ns, deadline_ns,
                           health_provider, &reservation, &next_frontier);
  if (!transaction.ok()) return transaction.status();
  auto status = executor_.submit_int4(
      std::move(*transaction), copy_driver, clear_driver, kernel_driver,
      head_driver, event_driver);
  if (!status.ok()) {
    poisoned_ = true;
    phase_ = QwenTeacherForcedPairRunDriverPhase::kPoisoned;
    return status;
  }
  if (!factory_capability_.has_value())
    factory_capability_ = factory.capability();
  commit(reservation, next_frontier);
  phase_ = QwenTeacherForcedPairRunDriverPhase::kInt4Pending;
  return Status::Ok();
}

Status QwenTeacherForcedPairRunDriver::poll(
    CompletionEventDriver& event_driver) {
  if (poisoned_)
    return Status::FailedPrecondition("Qwen paired run driver is poisoned");
  if (phase_ != QwenTeacherForcedPairRunDriverPhase::kBf16Pending &&
      phase_ != QwenTeacherForcedPairRunDriverPhase::kInt4Pending)
    return Status::FailedPrecondition(
        "Qwen paired run driver has no pending transaction");
  const auto previous = phase_;
  auto status = executor_.poll(event_driver);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    poisoned_ = true;
    phase_ = QwenTeacherForcedPairRunDriverPhase::kPoisoned;
  } else if (status.ok()) {
    phase_ = previous == QwenTeacherForcedPairRunDriverPhase::kBf16Pending
        ? QwenTeacherForcedPairRunDriverPhase::kAwaitingInt4
        : QwenTeacherForcedPairRunDriverPhase::kAwaitingBf16;
  }
  return status;
}

Status QwenTeacherForcedPairRunDriver::expire(std::uint64_t now_ns) {
  if (poisoned_)
    return Status::FailedPrecondition("Qwen paired run driver is poisoned");
  if (phase_ != QwenTeacherForcedPairRunDriverPhase::kBf16Pending &&
      phase_ != QwenTeacherForcedPairRunDriverPhase::kInt4Pending)
    return Status::FailedPrecondition(
        "Qwen paired run driver has no expirable transaction");
  auto status = executor_.expire(now_ns);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    poisoned_ = true;
    phase_ = QwenTeacherForcedPairRunDriverPhase::kPoisoned;
  }
  return status;
}

Result<std::array<QwenTeacherForcedCategoryAggregate, 6>>
QwenTeacherForcedPairRunDriver::finalize() {
  if (poisoned_)
    return Status::FailedPrecondition("Qwen paired run driver is poisoned");
  if (phase_ != QwenTeacherForcedPairRunDriverPhase::kAwaitingBf16)
    return Status::FailedPrecondition(
        "Qwen paired run driver has a partial pair");
  auto result = executor_.finalize();
  if (result.ok()) phase_ = QwenTeacherForcedPairRunDriverPhase::kFinalized;
  return result;
}

}  // namespace pih
