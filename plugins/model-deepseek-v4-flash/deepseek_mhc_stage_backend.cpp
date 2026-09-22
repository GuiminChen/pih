#include "pih/model/deepseek_mhc_stage_backend.h"

#include <unordered_set>

namespace pih {

Result<DeepSeekMhcStageOperatorBackend>
DeepSeekMhcStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& inner,
    DeepSeekMhcStageWorkProvider& provider) {
  DeepSeekMhcStageOperatorBackend value;
  value.inner_ = &inner;
  value.provider_ = &provider;
  return value;
}

Status DeepSeekMhcStageOperatorBackend::poison(Status status) {
  mode_ = Mode::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek mHC stage poisoned")
                     : status;
}

Status DeepSeekMhcStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0) {
    return Status::FailedPrecondition(
        "DeepSeek mHC stage backend is not launchable");
  }
  const bool wrapped = command.kind == DeepSeekStageOperatorKind::kAttention ||
                       command.kind == DeepSeekStageOperatorKind::kMoe;
  if (!wrapped) {
    auto status = inner_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kPassthrough;
    return Status::Ok();
  }
  auto resolved = provider_->resolve(command, plan);
  if (!resolved.ok()) return poison(resolved.status());
  if (resolved->size() != plan.sequence_count || resolved->empty()) {
    return poison(Status::InvalidArgument(
        "DeepSeek mHC work count does not match packed plan"));
  }
  const auto branch_kind = command.kind == DeepSeekStageOperatorKind::kAttention
                               ? DeepSeekMhcBranchKind::kAttention
                               : DeepSeekMhcBranchKind::kFeedForward;
  std::unordered_set<DeepSeekMhcSequenceExecutor*> executors;
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> transactions;
  for (const auto& item : *resolved) {
    if (item.executor == nullptr || item.transaction == nullptr ||
        item.submission.kind != branch_kind ||
        item.submission.layer_id != command.layer ||
        !executors.insert(item.executor).second ||
        !transactions.insert(item.transaction).second) {
      return poison(Status::InvalidArgument(
          "DeepSeek mHC packed work is invalid or duplicated"));
    }
  }
  work_.assign(resolved->begin(), resolved->end());
  for (auto& item : work_) {
    auto status = item.executor->begin(item.submission, *item.transaction);
    if (!status.ok()) return poison(status);
  }
  auto status = inner_->launch(command, plan);
  if (!status.ok()) return poison(status);
  mode_ = Mode::kMhc;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus> DeepSeekMhcStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek mHC stage backend is not pollable");
  }
  auto result = inner_->poll();
  if (!result.ok()) return poison(result.status());
  if (*result == DeepSeekStageComputeStatus::kInProgress) return *result;
  if (*result == DeepSeekStageComputeStatus::kError) {
    (void)poison(Status::Internal("DeepSeek mHC branch failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (mode_ == Mode::kMhc) {
    for (auto& item : work_) {
      auto status = item.executor->finish(*item.transaction);
      if (!status.ok()) return poison(status);
    }
    work_.clear();
  }
  mode_ = Mode::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
