#include "pih/model/deepseek_routed_stage_backend.h"

namespace pih {

Result<DeepSeekRoutedStageOperatorBackend>
DeepSeekRoutedStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& dense_backend,
    DeepSeekExpertPlanProvider& plan_provider, DeepSeekExpertPager& pager,
    DeepSeekExpertTransferDriver& transfer,
    DeepSeekExpertKernelDriver& kernel) {
  if (pager.poisoned()) {
    return Status::FailedPrecondition(
        "DeepSeek routed stage requires a healthy pager");
  }
  DeepSeekRoutedStageOperatorBackend backend;
  backend.dense_backend_ = &dense_backend;
  backend.plan_provider_ = &plan_provider;
  backend.pager_ = &pager;
  backend.transfer_ = &transfer;
  backend.kernel_ = &kernel;
  return backend;
}

Result<DeepSeekRoutedStageOperatorBackend>
DeepSeekRoutedStageOperatorBackend::CreateResident(
    DeepSeekStageOperatorBackend& dense_backend,
    DeepSeekExpertPlanProvider& plan_provider,
    DeepSeekExpertKernelDriver& kernel,
    const DeepSeekResidentExpertBindings& resident) {
  if (resident.generation() == 0) {
    return Status::InvalidArgument(
        "DeepSeek routed resident bindings are invalid");
  }
  DeepSeekRoutedStageOperatorBackend backend;
  backend.dense_backend_ = &dense_backend;
  backend.plan_provider_ = &plan_provider;
  backend.kernel_ = &kernel;
  backend.resident_ = &resident;
  return backend;
}

Status DeepSeekRoutedStageOperatorBackend::poison(Status status) {
  mode_ = Mode::kPoisoned;
  if (pager_ != nullptr) pager_->poison_epoch();
  return status.ok() ? Status::Internal("DeepSeek routed stage poisoned")
                     : status;
}

Status DeepSeekRoutedStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0) {
    return Status::FailedPrecondition(
        "DeepSeek routed stage backend is not launchable");
  }
  if (command.kind != DeepSeekStageOperatorKind::kMoe) {
    const auto status = dense_backend_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kDense;
    return Status::Ok();
  }
  auto expert_plan = plan_provider_->resolve(command.layer, plan);
  if (!expert_plan.ok() || *expert_plan == nullptr) {
    return poison(expert_plan.ok()
                      ? Status::FailedPrecondition(
                            "DeepSeek MoE route plan is unavailable")
                      : expert_plan.status());
  }
  if (command.layer > 42) {
    return poison(Status::InvalidArgument(
        "DeepSeek main routed backend rejects non-main layer identity"));
  }
  if (resident_ != nullptr) {
    auto executor = DeepSeekResidentSubwaveExecutor::Create(
        command.layer, **expert_plan, *resident_);
    if (!executor.ok()) return poison(executor.status());
    main_resident_executor_ = std::move(*executor);
    mode_ = Mode::kResidentMoe;
    const auto status = main_resident_executor_->advance(*kernel_);
    if (!status.ok()) return poison(status);
    return Status::Ok();
  }
  auto executor = DeepSeekExpertSubwaveExecutor::Create(
      static_cast<std::uint16_t>(command.layer), **expert_plan, *pager_);
  if (!executor.ok()) return poison(executor.status());
  executor_ = std::move(*executor);
  mode_ = Mode::kMoe;
  const auto status = executor_->advance(*transfer_, *kernel_);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    return poison(status);
  }
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekRoutedStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek routed stage backend is not pollable");
  }
  if (mode_ == Mode::kDense) {
    auto status = dense_backend_->poll();
    if (!status.ok()) return poison(status.status());
    if (*status == DeepSeekStageComputeStatus::kError) {
      (void)poison(Status::Internal("DeepSeek dense operator failed"));
      return DeepSeekStageComputeStatus::kError;
    }
    if (*status == DeepSeekStageComputeStatus::kSuccess) mode_ = Mode::kIdle;
    return *status;
  }
  if (mode_ == Mode::kResidentMoe) {
    const auto status = main_resident_executor_->advance(*kernel_);
    if (!status.ok()) return poison(status);
    if (main_resident_executor_->state() ==
        DeepSeekExpertSubwaveExecutorState::kPoisoned) {
      return poison(Status::Internal(
          "DeepSeek resident MoE operator failed"));
    }
    if (main_resident_executor_->state() !=
        DeepSeekExpertSubwaveExecutorState::kComplete) {
      return DeepSeekStageComputeStatus::kInProgress;
    }
    main_resident_executor_.reset();
    mode_ = Mode::kIdle;
    return DeepSeekStageComputeStatus::kSuccess;
  }
  const auto status = executor_->advance(*transfer_, *kernel_);
  if (!status.ok() && status.code() != StatusCode::kUnavailable) {
    return poison(status);
  }
  if (executor_->state() == DeepSeekExpertSubwaveExecutorState::kPoisoned) {
    (void)poison(Status::Internal("DeepSeek MoE operator failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (executor_->state() != DeepSeekExpertSubwaveExecutorState::kComplete) {
    return DeepSeekStageComputeStatus::kInProgress;
  }
  executor_.reset();
  mode_ = Mode::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
