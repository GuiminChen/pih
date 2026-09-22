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
    auto resolved = chunk_provider_->resolve(command.layer, plan);
    if (!resolved.ok() || *resolved == nullptr) {
      return poison(resolved.ok()
                        ? Status::FailedPrecondition(
                              "DeepSeek chunk attention work is unavailable")
                        : resolved.status());
    }
    const auto* chunk = *resolved;
    if (chunk->chunk_coordinator == nullptr ||
        chunk->attention_coordinator == nullptr ||
        chunk->transaction == nullptr) {
      return poison(Status::FailedPrecondition(
          "DeepSeek chunk attention work is incomplete"));
    }
    auto status = chunk->chunk_coordinator->launch(
        chunk->submission, *chunk->transaction);
    if (!status.ok()) return poison(status);
    active_coordinator_ = chunk->attention_coordinator;
    mode_ = Mode::kAttention;
    return Status::Ok();
  }
  if (command.kind != DeepSeekStageOperatorKind::kAttention ||
      plan.phase != DeepSeekPlanPhase::kDecode) {
    auto status = fallback_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kFallback;
    return Status::Ok();
  }
  auto resolved = provider_->resolve(command.layer, plan);
  if (!resolved.ok() || *resolved == nullptr) {
    return poison(resolved.ok()
                      ? Status::FailedPrecondition(
                            "DeepSeek decode attention work is unavailable")
                      : resolved.status());
  }
  work_ = *resolved;
  if (work_->recent_writer == nullptr || work_->update_coordinator == nullptr ||
      work_->coordinator == nullptr || work_->transaction == nullptr) {
    return poison(Status::FailedPrecondition(
        "DeepSeek decode attention work is incomplete"));
  }
  auto status = work_->coordinator->begin_decode(
      work_->recent, work_->update, work_->attention, *work_->recent_writer,
      *work_->update_coordinator, *work_->transaction);
  if (!status.ok()) return poison(status);
  active_coordinator_ = work_->coordinator;
  mode_ = Mode::kAttention;
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
  work_ = nullptr;
  active_coordinator_ = nullptr;
  mode_ = Mode::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
