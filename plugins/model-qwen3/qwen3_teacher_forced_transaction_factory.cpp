#include "pih/model/qwen3_teacher_forced_transaction_factory.h"

#include <array>
#include <utility>

namespace pih {
namespace {

bool valid_owner_ids(const QwenTeacherForcedMetricOwnerIds& value) {
  const std::array<std::uint64_t, 6> ids{
      value.pinned_sample_rows, value.device_sample_rows,
      value.pinned_targets, value.device_targets,
      value.device_result, value.pinned_result};
  for (std::size_t left = 0; left < ids.size(); ++left) {
    if (ids[left] == 0) return false;
    for (std::size_t right = left + 1; right < ids.size(); ++right)
      if (ids[left] == ids[right]) return false;
  }
  return true;
}

}  // namespace

Result<QwenTeacherForcedTransactionFactory>
QwenTeacherForcedTransactionFactory::Create(
    QwenTeacherForcedMetricArenas& arenas,
    ResolvedKernelFunction gather_function,
    ResolvedKernelFunction metric_function,
    TensorView normalized_rows, TensorView gathered_rows,
    TensorView lm_head_weight, TensorView logits,
    std::uintptr_t context_identity, DriverStreamHandle stream,
    std::int32_t owning_rank, QwenTeacherForcedMetricOwnerIds owner_ids) {
  if (context_identity == 0 || stream == 0 || owning_rank < 0 ||
      !valid_owner_ids(owner_ids) || normalized_rows.generation() == 0 ||
      normalized_rows.device().type() != DeviceType::kCuda ||
      normalized_rows.device().index() != owning_rank)
    return Status::InvalidArgument(
        "Qwen teacher-forced transaction factory identity is invalid");
  auto factory_generation = arenas.reserve_factory_generation();
  if (!factory_generation.ok()) return factory_generation.status();
  return QwenTeacherForcedTransactionFactory(
      arenas, std::move(gather_function), std::move(metric_function),
      normalized_rows, gathered_rows, lm_head_weight, logits,
      context_identity, stream, owning_rank, owner_ids,
      {arenas.arena_identity(), *factory_generation});
}

Result<QwenTeacherForcedChunkTransaction>
QwenTeacherForcedTransactionFactory::build(
    const QwenTeacherForcedBatchPlan& batch,
    std::uint64_t event_generation,
    std::uint64_t row_upload_plan_id,
    std::uint64_t target_upload_plan_id,
    std::uint64_t readback_plan_id,
    CompletionEventSlot completion_slot,
    CudaCompletionFrontier completion_frontier,
    QwenBf16StepHealthProvider& health_provider) {
  const auto rows = batch.sample_count();
  if (rows == 0 || batch.execution_tokens() != normalized_rows_.dim(0) ||
      static_cast<std::uint32_t>(gathered_rows_.dim(0)) != rows ||
      static_cast<std::uint32_t>(logits_.dim(0)) != rows ||
      event_generation == 0 ||
      completion_slot.state() != CompletionEventSlotState::kIdle ||
      completion_slot.context_identity() != context_identity_ ||
      completion_frontier.event_generation() != event_generation)
    return Status::InvalidArgument(
        "Qwen teacher-forced factory batch geometry differs");
  auto pinned_rows = arenas_->pinned_sample_rows_endpoint(
      owner_ids_.pinned_sample_rows);
  auto device_rows = arenas_->device_sample_rows_endpoint(
      owner_ids_.device_sample_rows);
  auto pinned_targets = arenas_->pinned_targets_endpoint(
      owner_ids_.pinned_targets);
  auto device_targets_endpoint = arenas_->device_targets_endpoint(
      owner_ids_.device_targets);
  auto device_result = arenas_->device_result_endpoint(owner_ids_.device_result);
  auto pinned_result = arenas_->pinned_result_endpoint(owner_ids_.pinned_result);
  if (!pinned_rows.ok()) return pinned_rows.status();
  if (!device_rows.ok()) return device_rows.status();
  if (!pinned_targets.ok()) return pinned_targets.status();
  if (!device_targets_endpoint.ok()) return device_targets_endpoint.status();
  if (!device_result.ok()) return device_result.status();
  if (!pinned_result.ok()) return pinned_result.status();
  auto transfer = QwenTeacherForcedMetricTransfer::Create(
      arenas_->layout(), rows, *pinned_rows, *device_rows, *pinned_targets,
      *device_targets_endpoint, *device_result, *pinned_result,
      context_identity_, stream_, event_generation, row_upload_plan_id,
      target_upload_plan_id, readback_plan_id);
  if (!transfer.ok()) return transfer.status();

  const auto generation = normalized_rows_.generation();
  auto sample_rows = arenas_->device_sample_rows(rows, generation);
  auto targets = arenas_->device_targets(rows, generation);
  auto argmax = arenas_->device_argmax(rows, generation);
  auto nll = arenas_->device_nll(rows, generation);
  auto nonfinite = arenas_->device_nonfinite(rows, generation);
  auto error = arenas_->device_error(generation);
  if (!sample_rows.ok()) return sample_rows.status();
  if (!targets.ok()) return targets.status();
  if (!argmax.ok()) return argmax.status();
  if (!nll.ok()) return nll.status();
  if (!nonfinite.ok()) return nonfinite.status();
  if (!error.ok()) return error.status();
  const std::array<std::int64_t, 1> error_bytes{4};
  auto wire_error = TensorView::Create(
      error->data(), DType::kUInt8, error_bytes, {}, error->device(), generation);
  if (!wire_error.ok()) return wire_error.status();
  auto logits_plan = QwenTeacherForcedLogitsPlan::Create(
      gather_function_, normalized_rows_, *sample_rows, gathered_rows_,
      lm_head_weight_, logits_, *wire_error, owning_rank_);
  if (!logits_plan.ok()) return logits_plan.status();
  auto metric_plan = QwenTeacherForcedMetricPlan::Create(
      metric_function_, logits_, *targets, *argmax, *nll, *nonfinite,
      *error, owning_rank_);
  if (!metric_plan.ok()) return metric_plan.status();
  auto arena_lease = arenas_->acquire_transaction_lease();
  if (!arena_lease.ok()) return arena_lease.status();
  auto status = batch.materialize_inputs(
      arenas_->pinned_sample_rows(), arenas_->pinned_targets());
  if (!status.ok()) return status;
  return QwenTeacherForcedChunkTransaction::Create(
      std::move(*transfer), std::move(*logits_plan), std::move(*metric_plan),
      *error, arenas_->layout(), std::move(completion_slot),
      std::move(completion_frontier), arenas_->pinned_result(), stream_,
      event_generation, owning_rank_, health_provider, std::move(*arena_lease));
}

Result<QwenTeacherForcedChunkTransaction>
QwenTeacherForcedTransactionFactory::build_reserved(
    const QwenTeacherForcedBatchPlan& batch,
    const QwenTeacherForcedTransactionIdentityReservation& identity,
    QwenTeacherForcedTransactionCompletion completion,
    QwenBf16StepHealthProvider& health_provider) {
  if (identity.row_upload_plan_id == 0 ||
      identity.target_upload_plan_id == 0 ||
      identity.readback_plan_id == 0 || identity.frontier_plan_id == 0 ||
      completion.frontier.key().plan_generation != identity.frontier_plan_id ||
      completion.frontier.key().epoch == 0 ||
      completion.frontier.key().rank !=
          static_cast<std::uint32_t>(owning_rank_) ||
      completion.frontier.key().phase != CudaCompletionPhase::kPrefill ||
      completion.frontier.key().completion_frontier == 0 ||
      completion.frontier.event_generation() != identity.event_generation)
    return Status::InvalidArgument(
        "Qwen teacher-forced reserved identity differs");
  return build(batch, identity.event_generation,
               identity.row_upload_plan_id, identity.target_upload_plan_id,
               identity.readback_plan_id, std::move(completion.slot),
               std::move(completion.frontier), health_provider);
}

}  // namespace pih
