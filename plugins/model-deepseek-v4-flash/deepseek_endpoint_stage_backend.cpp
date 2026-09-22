#include "pih/model/deepseek_endpoint_stage_backend.h"

#include <unordered_set>

namespace pih {

Result<DeepSeekEndpointStageOperatorBackend>
DeepSeekEndpointStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& fallback,
    DeepSeekEndpointStageWorkProvider& provider) {
  DeepSeekEndpointStageOperatorBackend value;
  value.fallback_ = &fallback;
  value.provider_ = &provider;
  return value;
}

Status DeepSeekEndpointStageOperatorBackend::poison(Status status) {
  mode_ = Mode::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek endpoint stage poisoned")
                     : status;
}

Status DeepSeekEndpointStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint stage backend is not launchable");
  }
  const bool endpoint = command.kind == DeepSeekStageOperatorKind::kEmbedding ||
                        command.kind == DeepSeekStageOperatorKind::kHead;
  if (!endpoint) {
    auto status = fallback_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kFallback;
    return Status::Ok();
  }
  auto resolved = provider_->resolve(command, plan);
  if (!resolved.ok()) return poison(resolved.status());
  if (resolved->empty() || resolved->size() != plan.sequence_count) {
    return poison(Status::InvalidArgument(
        "DeepSeek endpoint work count does not match packed plan"));
  }
  std::unordered_set<DeepSeekEndpointSequenceExecutor*> executors;
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> transactions;
  for (const auto& item : *resolved) {
    if (item.executor == nullptr || item.transaction == nullptr ||
        !executors.insert(item.executor).second ||
        !transactions.insert(item.transaction).second) {
      return poison(Status::InvalidArgument(
          "DeepSeek endpoint packed work is invalid or duplicated"));
    }
  }
  for (const auto& item : *resolved) {
    auto status = command.kind == DeepSeekStageOperatorKind::kEmbedding
        ? item.executor->launch_embedding(item.embedding, *item.transaction)
        : item.executor->launch_head(item.head, *item.transaction);
    if (!status.ok()) return poison(status);
  }
  mode_ = Mode::kEndpoint;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekEndpointStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint stage backend is not pollable");
  }
  if (mode_ == Mode::kEndpoint) {
    mode_ = Mode::kIdle;
    return DeepSeekStageComputeStatus::kSuccess;
  }
  auto result = fallback_->poll();
  if (!result.ok()) return poison(result.status());
  if (*result == DeepSeekStageComputeStatus::kError) {
    (void)poison(Status::Internal("DeepSeek endpoint fallback failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (*result == DeepSeekStageComputeStatus::kSuccess) mode_ = Mode::kIdle;
  return *result;
}

}  // namespace pih
