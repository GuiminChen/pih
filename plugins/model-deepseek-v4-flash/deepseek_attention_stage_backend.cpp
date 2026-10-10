#include "pih/model/deepseek_attention_stage_backend.h"

namespace pih {

Result<DeepSeekAttentionStageOperatorBackend>
DeepSeekAttentionStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& fallback,
    DeepSeekDecodeAttentionWorkProvider& provider) {
  DeepSeekAttentionStageOperatorBackend backend;
  backend.fallback_ = &fallback;
  backend.provider_ = &provider;
  return backend;
}

Result<DeepSeekAttentionStageOperatorBackend>
DeepSeekAttentionStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& fallback,
    DeepSeekDecodeAttentionWorkProvider& decode_provider,
    DeepSeekChunkAttentionWorkProvider& chunk_provider) {
  auto backend = Create(fallback, decode_provider);
  if (!backend.ok()) return backend.status();
  backend->chunk_provider_ = &chunk_provider;
  return backend;
}

Status DeepSeekAttentionStageOperatorBackend::poison(Status status) {
  mode_ = Mode::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek attention stage poisoned")
                     : status;
}

Status DeepSeekAttentionStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0) {
    return Status::FailedPrecondition(
        "DeepSeek attention stage backend is not launchable");
  }
  if (command.kind == DeepSeekStageOperatorKind::kAttention &&
      (plan.phase == DeepSeekPlanPhase::kPrefill ||
       plan.phase == DeepSeekPlanPhase::kVerify) &&
      chunk_provider_ != nullptr) {
    auto resolved = chunk_provider_->resolve_batch(command.layer, plan);
    if (!resolved.ok() || resolved->empty())
      return poison(resolved.ok() ? Status::FailedPrecondition("DeepSeek packed prefill work missing") : resolved.status());
    chunk_batch_ = *resolved;
    decode_batch_ = {};
    sequence_cursor_ = 0;
    mode_ = Mode::kAttention;
    return launch_sequence();
  }
  if (command.kind != DeepSeekStageOperatorKind::kAttention ||
      plan.phase != DeepSeekPlanPhase::kDecode) {
    auto status = fallback_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kFallback;
    return Status::Ok();
  }
  auto resolved = provider_->resolve_batch(command.layer, plan);
  if (!resolved.ok() || resolved->empty())
    return poison(resolved.ok() ? Status::FailedPrecondition("DeepSeek packed decode work missing") : resolved.status());
  decode_batch_ = *resolved;
  chunk_batch_ = {};
  sequence_cursor_ = 0;
  mode_ = Mode::kAttention;
  return launch_sequence();
}

Status DeepSeekAttentionStageOperatorBackend::launch_sequence() {
  if (!chunk_batch_.empty()) {
    const auto& chunk = chunk_batch_[sequence_cursor_];
    if (!chunk.chunk_coordinator || !chunk.attention_coordinator || !chunk.transaction)
      return poison(Status::FailedPrecondition("DeepSeek packed prefill resources incomplete"));
    auto status = chunk.chunk_coordinator->launch(chunk.submission, *chunk.transaction);
    if (!status.ok()) return poison(status);
    active_coordinator_ = chunk.attention_coordinator;
  } else {
    work_ = &decode_batch_[sequence_cursor_];
    if (!work_->recent_writer || !work_->update_coordinator || !work_->coordinator || !work_->transaction)
      return poison(Status::FailedPrecondition("DeepSeek packed decode resources incomplete"));
    auto status = work_->coordinator->begin_decode(work_->recent, work_->update,
        work_->attention, *work_->recent_writer, *work_->update_coordinator, *work_->transaction);
    if (!status.ok()) return poison(status);
    active_coordinator_ = work_->coordinator;
  }
  return Status::Ok();
}
Result<DeepSeekStageComputeStatus>
DeepSeekAttentionStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek attention stage backend is not pollable");
  }
  if (mode_ == Mode::kFallback) {
    auto status = fallback_->poll();
    if (!status.ok()) return poison(status.status());
    if (*status == DeepSeekStageComputeStatus::kError) {
      (void)poison(Status::Internal("DeepSeek fallback operator failed"));
      return DeepSeekStageComputeStatus::kError;
    }
    if (*status == DeepSeekStageComputeStatus::kSuccess) mode_ = Mode::kIdle;
    return *status;
  }
  if (active_coordinator_ == nullptr) {
    return poison(Status::Internal(
        "DeepSeek attention stage lost its active coordinator"));
  }
  switch (active_coordinator_->state()) {
    case DeepSeekAttentionLayerCoordinatorState::kSelectionReady: {
      auto status = active_coordinator_->launch_next_selection_tile();
      if (!status.ok()) return poison(status);
      return DeepSeekStageComputeStatus::kInProgress;
    }
    case DeepSeekAttentionLayerCoordinatorState::kSelectionInflight: {
      auto status = active_coordinator_->poll_selection_tile();
      if (!status.ok()) return poison(status.status());
      if (*status == DeepSeekExpertAsyncStatus::kError) {
        (void)poison(Status::Internal("DeepSeek attention selection failed"));
        return DeepSeekStageComputeStatus::kError;
      }
      if (active_coordinator_->state() !=
          DeepSeekAttentionLayerCoordinatorState::kAttentionPosted) {
        return DeepSeekStageComputeStatus::kInProgress;
      }
      break;
    }
    case DeepSeekAttentionLayerCoordinatorState::kAttentionPosted:
      break;
    default:
      return poison(Status::Internal(
          "DeepSeek attention coordinator entered an invalid stage state"));
  }
  if (++sequence_cursor_ < (chunk_batch_.empty() ? decode_batch_.size() : chunk_batch_.size())) {
    auto status = launch_sequence();
    if (!status.ok()) return status;
    return DeepSeekStageComputeStatus::kInProgress;
  }
  decode_batch_ = {};
  chunk_batch_ = {};
  work_ = nullptr;
  active_coordinator_ = nullptr;
  mode_ = Mode::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
