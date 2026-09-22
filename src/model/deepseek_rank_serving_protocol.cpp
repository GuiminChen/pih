#include "pih/model/deepseek_rank_serving_protocol.h"

#include <algorithm>
#include <limits>

#include "pih/model/deepseek_rank_materialization_warmup.h"
#include "pih/model/deepseek_rank_process_supervisor.h"

namespace pih {
namespace {

bool digest_is_zero(const Sha256Digest& digest) {
  return std::all_of(digest.bytes.begin(), digest.bytes.end(),
                     [](std::byte value) { return value == std::byte{0}; });
}

bool valid_phase(DeepSeekPlanPhase phase) {
  switch (phase) {
    case DeepSeekPlanPhase::kPrefill:
    case DeepSeekPlanPhase::kDecode:
    case DeepSeekPlanPhase::kVerify:
      return true;
    case DeepSeekPlanPhase::kDrain:
      return false;
  }
  return false;
}

bool same_plan(const DeepSeekPipelinePlanDescriptor& lhs,
               const DeepSeekPipelinePlanDescriptor& rhs) {
  return lhs.engine_epoch == rhs.engine_epoch &&
         lhs.plan_sequence == rhs.plan_sequence && lhs.phase == rhs.phase &&
         lhs.token_count == rhs.token_count &&
         lhs.sequence_count == rhs.sequence_count;
}

Status validate_session_binding(const DeepSeekRankServingSessionBinding& session) {
  if (session.engine_epoch == 0 || session.worker_generation == 0 ||
      session.world_size == 0 || session.rank >= session.world_size ||
      session.process_identity == 0 || session.pidfd_identity == 0 ||
      session.control_identity == 0 ||
      digest_is_zero(session.materialization_warm_seal_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank serving session binding is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekRankServingSessionBinding>
compile_deepseek_rank_serving_session_binding(
    const DeepSeekRankReadyReceipt& ready,
    const DeepSeekRankMaterializationWarmSeal& warm_seal) {
  if (!warm_seal.production_eligible() || ready.engine_epoch == 0 ||
      ready.engine_epoch != warm_seal.engine_epoch() ||
      ready.worker_generation == 0 ||
      ready.worker_generation != warm_seal.worker_generation() ||
      ready.rank >= warm_seal.world_size()) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving binding does not match materialization seal");
  }
  DeepSeekRankServingSessionBinding binding{
      ready.engine_epoch,
      ready.worker_generation,
      warm_seal.world_size(),
      ready.rank,
      ready.process_identity,
      ready.pidfd_identity,
      ready.control_identity,
      warm_seal.seal_root(),
  };
  const auto status = validate_session_binding(binding);
  if (!status.ok()) return status;
  return binding;
}

Result<std::vector<DeepSeekRankServingSessionBinding>>
compile_deepseek_rank_serving_session_set(
    std::span<const DeepSeekRankReadyReceipt> ready_receipts,
    const DeepSeekRankMaterializationWarmSeal& warm_seal) {
  if (ready_receipts.empty() || ready_receipts.size() > 4 ||
      ready_receipts.size() != warm_seal.world_size()) {
    return Status::InvalidArgument(
        "DeepSeek rank serving session set geometry is invalid");
  }
  std::vector<DeepSeekRankServingSessionBinding> sessions;
  sessions.reserve(ready_receipts.size());
  for (std::uint32_t rank = 0; rank < ready_receipts.size(); ++rank) {
    if (ready_receipts[rank].rank != rank) {
      return Status::FailedPrecondition(
          "DeepSeek rank serving ready receipts are not rank ordered");
    }
    auto session = compile_deepseek_rank_serving_session_binding(
        ready_receipts[rank], warm_seal);
    if (!session.ok()) return session.status();
    sessions.push_back(std::move(*session));
  }
  return sessions;
}

Status DeepSeekRankServingWorkerGate::ValidateSession(
    const DeepSeekRankServingSessionBinding& session) {
  return validate_session_binding(session);
}

Result<DeepSeekRankServingWorkerGate> DeepSeekRankServingWorkerGate::Create(
    DeepSeekRankServingSessionBinding session) {
  const auto status = ValidateSession(session);
  if (!status.ok()) return status;
  return DeepSeekRankServingWorkerGate(std::move(session));
}

Status DeepSeekRankServingWorkerGate::Reject(Status cause) noexcept {
  failed_ = true;
  return cause.ok()
             ? Status::Internal("DeepSeek rank serving protocol rejected")
             : cause;
}

Status DeepSeekRankServingWorkerGate::ValidateEnvelope(
    const DeepSeekRankServingCommand& command) const {
  if (command.session != session_ || command.command_sequence == 0 ||
      last_command_sequence_ == std::numeric_limits<std::uint64_t>::max() ||
      command.command_sequence != last_command_sequence_ + 1 ||
      command.request_id == 0 || command.request_generation == 0 ||
      command.plan.engine_epoch != session_.engine_epoch ||
      command.plan.plan_sequence == 0 || !valid_phase(command.plan.phase) ||
      command.plan.token_count == 0 || command.plan.sequence_count == 0) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving command envelope does not match session");
  }
  return Status::Ok();
}

Status DeepSeekRankServingWorkerGate::accept(
    const DeepSeekRankServingCommand& command) {
  if (failed_) {
    return Status::FailedPrecondition("DeepSeek rank serving gate is failed");
  }
  const auto envelope = ValidateEnvelope(command);
  if (!envelope.ok()) return Reject(envelope);

  if (command.kind == DeepSeekRankServingCommandKind::kExecute) {
    if (active_execution_ || command.target_execution_sequence != 0 ||
        (session_.rank == 0) != (command.input_lease_identity != 0) ||
        (session_.rank + 1 == session_.world_size) !=
            (command.output_lease_identity != 0)) {
      return Reject(Status::FailedPrecondition(
          "DeepSeek rank execute command violates ownership or ordering"));
    }
    last_command_sequence_ = command.command_sequence;
    active_execution_ = command;
    return Status::Ok();
  }

  if (command.kind == DeepSeekRankServingCommandKind::kCancel) {
    if (!active_execution_ || cancellation_command_sequence_ ||
        command.input_lease_identity != 0 || command.output_lease_identity != 0 ||
        command.target_execution_sequence !=
            active_execution_->command_sequence ||
        command.request_id != active_execution_->request_id ||
        command.request_generation != active_execution_->request_generation ||
        !same_plan(command.plan, active_execution_->plan)) {
      return Reject(Status::FailedPrecondition(
          "DeepSeek rank cancellation does not match active execution"));
    }
    last_command_sequence_ = command.command_sequence;
    cancellation_command_sequence_ = command.command_sequence;
    return Status::Ok();
  }
  return Reject(Status::InvalidArgument("DeepSeek rank serving command kind is invalid"));
}

Status DeepSeekRankServingWorkerGate::complete(
    const DeepSeekRankServingCompletion& completion) {
  if (failed_) {
    return Status::FailedPrecondition("DeepSeek rank serving gate is failed");
  }
  if (!active_execution_ || completion.session != session_ ||
      completion.execution_command_sequence !=
          active_execution_->command_sequence ||
      completion.request_id != active_execution_->request_id ||
      completion.request_generation != active_execution_->request_generation ||
      !same_plan(completion.plan, active_execution_->plan)) {
    return Reject(Status::FailedPrecondition(
        "DeepSeek rank completion does not match active execution"));
  }

  const bool cancellation_requested = cancellation_command_sequence_.has_value();
  if (completion.outcome == DeepSeekRankServingCompletionOutcome::kCompleted) {
    if (cancellation_requested ||
        completion.output_lease_identity !=
            active_execution_->output_lease_identity) {
      return Reject(Status::FailedPrecondition(
          "DeepSeek rank completion violates cancellation or output ownership"));
    }
  } else if (completion.outcome == DeepSeekRankServingCompletionOutcome::kCancelled) {
    if (!cancellation_requested || completion.output_lease_identity != 0) {
      return Reject(Status::FailedPrecondition(
          "DeepSeek rank cancellation completion is invalid"));
    }
  } else if (completion.outcome == DeepSeekRankServingCompletionOutcome::kFailed) {
    if (completion.output_lease_identity != 0) {
      return Reject(Status::FailedPrecondition(
          "DeepSeek rank failed completion cannot publish output"));
    }
  } else {
    return Reject(Status::InvalidArgument(
        "DeepSeek rank completion outcome is invalid"));
  }

  active_execution_.reset();
  cancellation_command_sequence_.reset();
  return Status::Ok();
}

}  // namespace pih
