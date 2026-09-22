#include "pih/model/deepseek_rank_serving_plan_executor.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& lhs,
                     const DeepSeekPipelinePlanDescriptor& rhs) {
  return lhs.engine_epoch == rhs.engine_epoch &&
         lhs.plan_sequence == rhs.plan_sequence && lhs.phase == rhs.phase &&
         lhs.token_count == rhs.token_count &&
         lhs.sequence_count == rhs.sequence_count;
}

}  // namespace

Result<DeepSeekRankPlanRuntimeServingExecution>
DeepSeekRankPlanRuntimeServingExecution::Create(
    DeepSeekRankPlanRuntime runtime,
    const DeepSeekRankServingCommand& command) {
  if (command.kind != DeepSeekRankServingCommandKind::kExecute ||
      runtime.state() != DeepSeekRankPlanRuntimeState::kCommitted ||
      runtime.rank() != command.session.rank ||
      runtime.world_size() != command.session.world_size ||
      command.plan.engine_epoch != command.session.engine_epoch ||
      !same_descriptor(runtime.descriptor(), command.plan)) {
    return Status::FailedPrecondition(
        "DeepSeek serving command differs from committed rank runtime");
  }
  return DeepSeekRankPlanRuntimeServingExecution(std::move(runtime));
}

Status DeepSeekRankPlanRuntimeServingExecution::cancel() {
  return runtime_.cancel();
}

Result<bool> DeepSeekRankPlanRuntimeServingExecution::advance() {
  auto status = runtime_.advance();
  if (status.code() == StatusCode::kUnavailable) return false;
  if (!status.ok()) return status;
  return runtime_.state() == DeepSeekRankPlanRuntimeState::kComplete;
}

Result<DeepSeekRankServingPlanWorkerExecutor>
DeepSeekRankServingPlanWorkerExecutor::Create(
    DeepSeekRankServingPlanExecutionFactory& factory,
    DeepSeekRankServingLeaseResolver& resolver) {
  return DeepSeekRankServingPlanWorkerExecutor(factory, resolver);
}

Status DeepSeekRankServingPlanWorkerExecutor::start(
    const DeepSeekRankServingCommand& command) {
  if (execution_ != nullptr || active_command_.has_value() ||
      command.kind != DeepSeekRankServingCommandKind::kExecute) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving plan executor already has an execution");
  }
  auto leases = resolver_->resolve(command);
  if (!leases.ok()) return leases.status();
  if (leases->input_lease_identity != command.input_lease_identity ||
      leases->output_lease_identity != command.output_lease_identity ||
      (command.input_lease_identity != 0 && leases->input_owner == nullptr) ||
      (command.output_lease_identity != 0 && leases->output_owner == nullptr)) {
    return Status::FailedPrecondition("DeepSeek serving lease resolution differs from command");
  }
  auto execution = factory_->start(command, std::move(*leases));
  if (!execution.ok()) return execution.status();
  if (*execution == nullptr) {
    return Status::Internal(
        "DeepSeek rank serving plan factory returned no execution");
  }
  execution_ = std::move(*execution);
  active_command_ = command;
  cancellation_requested_ = false;
  return Status::Ok();
}

Status DeepSeekRankServingPlanWorkerExecutor::cancel(
    const DeepSeekRankServingCommand& command) {
  if (execution_ == nullptr || !active_command_.has_value() ||
      cancellation_requested_ ||
      command.kind != DeepSeekRankServingCommandKind::kCancel ||
      command.target_execution_sequence !=
          active_command_->command_sequence ||
      command.input_lease_identity != 0 || command.output_lease_identity != 0 ||
      command.session != active_command_->session ||
      command.request_id != active_command_->request_id ||
      command.request_generation != active_command_->request_generation ||
      command.plan.engine_epoch != active_command_->plan.engine_epoch ||
      command.plan.plan_sequence != active_command_->plan.plan_sequence ||
      command.plan.phase != active_command_->plan.phase ||
      command.plan.token_count != active_command_->plan.token_count ||
      command.plan.sequence_count != active_command_->plan.sequence_count) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving cancellation differs from active execution");
  }
  auto status = execution_->cancel();
  if (!status.ok()) return status;
  cancellation_requested_ = true;
  return Status::Ok();
}

Result<std::optional<DeepSeekRankServingCompletion>>
DeepSeekRankServingPlanWorkerExecutor::poll() {
  if (execution_ == nullptr || !active_command_.has_value()) {
    return std::optional<DeepSeekRankServingCompletion>{};
  }
  auto terminal = execution_->advance();
  if (!terminal.ok()) return terminal.status();
  if (!*terminal) return std::optional<DeepSeekRankServingCompletion>{};

  DeepSeekRankServingCompletion completion;
  completion.session = active_command_->session;
  completion.execution_command_sequence = active_command_->command_sequence;
  completion.request_id = active_command_->request_id;
  completion.request_generation = active_command_->request_generation;
  completion.plan = active_command_->plan;
  completion.outcome = cancellation_requested_
      ? DeepSeekRankServingCompletionOutcome::kCancelled
      : DeepSeekRankServingCompletionOutcome::kCompleted;
  completion.output_lease_identity = cancellation_requested_
      ? 0
      : active_command_->output_lease_identity;

  execution_.reset();
  active_command_.reset();
  cancellation_requested_ = false;
  return std::optional<DeepSeekRankServingCompletion>(std::move(completion));
}

}  // namespace pih
