#include "pih/model/deepseek_router_stage_backend.h"

namespace pih {

Result<DeepSeekRouterStageOperatorBackend>
DeepSeekRouterStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& routed_backend,
    DeepSeekRouterStageWorkProvider& work_provider) {
  if (work_provider.plan_provider() == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek router requires an explicit expert plan provider");
  }
  DeepSeekRouterStageOperatorBackend result;
  result.routed_backend_ = &routed_backend;
  result.work_provider_ = &work_provider;
  return result;
}

Status DeepSeekRouterStageOperatorBackend::poison(Status status) {
  mode_ = Mode::kPoisoned;
  pending_command_.reset();
  pending_plan_.reset();
  learned_coordinator_ = nullptr;
  hash_coordinator_ = nullptr;
  return status.ok() ? Status::Internal("DeepSeek router stage poisoned")
                     : status;
}

Status DeepSeekRouterStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle) {
    return Status::FailedPrecondition(
        "DeepSeek router stage is busy or poisoned");
  }
  if (command.kind != DeepSeekStageOperatorKind::kMoe) {
    const auto status = routed_backend_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kInner;
    return Status::Ok();
  }
  if (plan.token_count == 0 || command.layer >= 43) {
    return poison(Status::InvalidArgument(
        "DeepSeek router stage identity is invalid"));
  }
  if (command.layer < 3) {
    auto work = work_provider_->resolve_hash(command, plan);
    if (!work.ok()) return poison(work.status());
    if (work->coordinator != nullptr) {
      if (work->coordinator->plan_provider() !=
              work_provider_->plan_provider() ||
          work->submission.projection.layer != command.layer ||
          work->submission.projection.token_count != plan.token_count) {
        return poison(Status::FailedPrecondition(
            "DeepSeek projected hash router stage work is invalid"));
      }
      const auto status = work->coordinator->launch(work->submission);
      if (!status.ok()) return poison(status);
      hash_coordinator_ = work->coordinator;
      pending_command_ = command;
      pending_plan_ = plan;
      mode_ = Mode::kHashRouter;
      return Status::Ok();
    }
    if (work->token_ids.size() != plan.token_count ||
        work->scratch == nullptr || work->store == nullptr ||
        static_cast<DeepSeekExpertPlanProvider*>(work->store) !=
            work_provider_->plan_provider()) {
      return poison(Status::FailedPrecondition(
          "DeepSeek hash router stage work is invalid"));
    }
    const auto routed = DeepSeekHashRouter::RouteInto(
        command.layer, work->token_ids, work->raw_scores,
        work->vocabulary_size, work->token_to_experts, *work->scratch,
        *work->store);
    if (!routed.ok()) return poison(routed);
    const auto status = routed_backend_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kInner;
    return Status::Ok();
  }

  auto work = work_provider_->resolve_learned(command, plan);
  if (!work.ok()) return poison(work.status());
  if (work->coordinator == nullptr ||
      work->coordinator->plan_provider() != work_provider_->plan_provider() ||
      work->submission.layer != command.layer ||
      work->submission.token_count != plan.token_count) {
    return poison(Status::FailedPrecondition(
        "DeepSeek learned router stage work is invalid"));
  }
  const auto status = work->coordinator->launch(work->submission);
  if (!status.ok()) return poison(status);
  learned_coordinator_ = work->coordinator;
  pending_command_ = command;
  pending_plan_ = plan;
  mode_ = Mode::kLearnedRouter;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekRouterStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek router stage is not pollable");
  }
  if (mode_ == Mode::kLearnedRouter) {
    auto routed = learned_coordinator_->poll();
    if (!routed.ok()) return poison(routed.status());
    if (*routed == DeepSeekStageComputeStatus::kInProgress) return *routed;
    if (*routed == DeepSeekStageComputeStatus::kError) {
      mode_ = Mode::kPoisoned;
      return *routed;
    }
    const auto status = routed_backend_->launch(*pending_command_, *pending_plan_);
    if (!status.ok()) return poison(status);
    pending_command_.reset();
    pending_plan_.reset();
    learned_coordinator_ = nullptr;
    mode_ = Mode::kInner;
    return DeepSeekStageComputeStatus::kInProgress;
  }
  if (mode_ == Mode::kHashRouter) {
    auto routed = hash_coordinator_->poll();
    if (!routed.ok()) return poison(routed.status());
    if (*routed == DeepSeekStageComputeStatus::kInProgress) return *routed;
    if (*routed == DeepSeekStageComputeStatus::kError) {
      mode_ = Mode::kPoisoned;
      return *routed;
    }
    const auto status = routed_backend_->launch(*pending_command_, *pending_plan_);
    if (!status.ok()) return poison(status);
    pending_command_.reset();
    pending_plan_.reset();
    hash_coordinator_ = nullptr;
    mode_ = Mode::kInner;
    return DeepSeekStageComputeStatus::kInProgress;
  }

  auto status = routed_backend_->poll();
  if (!status.ok()) return poison(status.status());
  if (*status == DeepSeekStageComputeStatus::kError) {
    mode_ = Mode::kPoisoned;
  } else if (*status == DeepSeekStageComputeStatus::kSuccess) {
    mode_ = Mode::kIdle;
  }
  return *status;
}

}  // namespace pih
