#include "pih/model/deepseek_dspark_stage_backend.h"

namespace pih {

Result<DeepSeekDsparkStageOperatorBackend>
DeepSeekDsparkStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& fallback,
    DeepSeekDsparkStageWorkProvider& provider,
    DeepSeekDsparkBlockExecutor& block_executor) {
  DeepSeekDsparkStageOperatorBackend value;
  value.fallback_ = &fallback;
  value.provider_ = &provider;
  value.block_executor_ = &block_executor;
  return value;
}

Status DeepSeekDsparkStageOperatorBackend::rollback_work() noexcept {
  if (work_ == nullptr || work_->transaction == nullptr) return Status::Ok();
  const auto state = work_->transaction->state();
  if (state == DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return work_->transaction->cancel_preparing();
  }
  if (state == DeepSeekAttentionSequenceTransactionState::kReadyToResolve) {
    return work_->transaction->abort();
  }
  if (state == DeepSeekAttentionSequenceTransactionState::kIdle) {
    return Status::Ok();
  }
  return Status::FailedPrecondition(
      "DeepSeek DSpark transaction cannot be rolled back in its current state");
}

Status DeepSeekDsparkStageOperatorBackend::poison(Status status) {
  if (work_ != nullptr && next_stage_ < evidence_.size() &&
      evidence_[next_stage_].state ==
          DeepSeekDsparkStageEvidenceState::kInFlight) {
    evidence_[next_stage_].state = DeepSeekDsparkStageEvidenceState::kFailed;
  }
  const auto rollback = rollback_work();
  work_ = nullptr;
  mode_ = Mode::kPoisoned;
  if (!rollback.ok()) return rollback;
  return status.ok() ? Status::Internal("DeepSeek DSpark stage poisoned")
                     : status;
}

Status DeepSeekDsparkStageOperatorBackend::launch_stage(
    std::uint32_t index) {
  auto stage = deepseek_dspark_stage_id(index);
  if (!stage.ok()) return stage.status();
  next_stage_ = index;
  evidence_[index].state = DeepSeekDsparkStageEvidenceState::kInFlight;
  return block_executor_->launch(*stage, plan_);
}

Status DeepSeekDsparkStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0 || plan.token_count == 0 ||
      plan.sequence_count == 0) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark stage backend is not launchable");
  }
  if (command.kind != DeepSeekStageOperatorKind::kDspark) {
    auto status = fallback_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kFallback;
    return Status::Ok();
  }
  if (plan.phase != DeepSeekPlanPhase::kPrefill &&
      plan.phase != DeepSeekPlanPhase::kDecode) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark orchestration phase is invalid");
  }
  auto resolved = provider_->resolve(plan);
  if (!resolved.ok() || *resolved == nullptr) {
    return poison(resolved.ok()
        ? Status::FailedPrecondition("DeepSeek DSpark stage work is unavailable")
        : resolved.status());
  }
  work_ = *resolved;
  const auto expected_kind = plan.phase == DeepSeekPlanPhase::kPrefill
      ? DeepSeekDsparkStageWorkKind::kPrefillStateInitialization
      : DeepSeekDsparkStageWorkKind::kDecodeProposal;
  if (work_->kind != expected_kind || work_->embed_coordinator == nullptr ||
      work_->transaction == nullptr ||
      (plan.phase == DeepSeekPlanPhase::kDecode &&
       work_->head_executor == nullptr)) {
    return poison(Status::InvalidArgument(
        "DeepSeek DSpark stage work is incomplete"));
  }
  plan_ = plan;
  head_submitted_ = false;
  prefill_state_initialized_ = false;
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    evidence_[index] = {
        deepseek_dspark_stage_id(index).value(),
        DeepSeekDsparkStageEvidenceState::kNotStarted,
        plan.engine_epoch,
        plan.plan_sequence};
  }
  auto status = plan.phase == DeepSeekPlanPhase::kPrefill
      ? work_->embed_coordinator->launch_prefill_state(
            work_->embed, *work_->transaction)
      : work_->embed_coordinator->launch(work_->embed,
                                         *work_->transaction);
  if (!status.ok()) return poison(status);
  status = launch_stage(0);
  if (!status.ok()) return poison(status);
  mode_ = Mode::kBlocks;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus> DeepSeekDsparkStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark stage backend is not pollable");
  }
  if (mode_ == Mode::kFallback) {
    auto result = fallback_->poll();
    if (!result.ok()) return poison(result.status());
    if (*result == DeepSeekStageComputeStatus::kError) {
      (void)poison(Status::Internal("DeepSeek DSpark fallback failed"));
      return DeepSeekStageComputeStatus::kError;
    }
    if (*result == DeepSeekStageComputeStatus::kSuccess) mode_ = Mode::kIdle;
    return *result;
  }
  auto result = block_executor_->poll();
  if (!result.ok()) return poison(result.status());
  if (*result == DeepSeekStageComputeStatus::kInProgress) return *result;
  if (*result == DeepSeekStageComputeStatus::kError) {
    evidence_[next_stage_].state = DeepSeekDsparkStageEvidenceState::kFailed;
    const auto rollback = rollback_work();
    work_ = nullptr;
    mode_ = Mode::kPoisoned;
    if (!rollback.ok()) return rollback;
    return DeepSeekStageComputeStatus::kError;
  }
  evidence_[next_stage_].state =
      DeepSeekDsparkStageEvidenceState::kSucceeded;
  if (next_stage_ + 1 < kDeepSeekDsparkStageCount) {
    const auto status = launch_stage(next_stage_ + 1);
    if (!status.ok()) return poison(status);
    return DeepSeekStageComputeStatus::kInProgress;
  }
  if (plan_.phase == DeepSeekPlanPhase::kPrefill) {
    prefill_state_initialized_ = true;
    work_ = nullptr;
    mode_ = Mode::kIdle;
    return DeepSeekStageComputeStatus::kSuccess;
  }
  auto status = work_->head_executor->launch(work_->head, *work_->transaction);
  if (!status.ok()) return poison(status);
  head_submitted_ = true;
  work_ = nullptr;
  mode_ = Mode::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

Status DeepSeekDsparkStageOperatorBackend::cancel() {
  if (mode_ != Mode::kBlocks || work_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark stage backend has no cancellable transaction");
  }
  if (next_stage_ < evidence_.size() &&
      evidence_[next_stage_].state ==
          DeepSeekDsparkStageEvidenceState::kInFlight) {
    evidence_[next_stage_].state =
        DeepSeekDsparkStageEvidenceState::kCancelled;
  }
  const auto cancel = block_executor_->cancel();
  const auto rollback = rollback_work();
  work_ = nullptr;
  mode_ = Mode::kPoisoned;
  if (!cancel.ok()) return cancel;
  return rollback;
}

}  // namespace pih
