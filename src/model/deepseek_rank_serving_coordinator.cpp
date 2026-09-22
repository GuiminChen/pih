#include "pih/model/deepseek_rank_serving_coordinator.h"

#include <limits>
#include <utility>

namespace pih {

Result<DeepSeekRankServingCoordinator> DeepSeekRankServingCoordinator::Create(
    std::span<const DeepSeekRankServingSessionBinding> sessions,
    DeepSeekRankServingChannel& channel) {
  if (sessions.empty() || sessions.size() > 4) {
    return Status::InvalidArgument("DeepSeek rank serving session geometry is invalid");
  }
  std::vector<DeepSeekRankServingSessionBinding> copied(sessions.begin(),
                                                        sessions.end());
  std::vector<DeepSeekRankServingWorkerGate> gates;
  gates.reserve(copied.size());
  const auto& first = copied.front();
  for (std::uint32_t rank = 0; rank < copied.size(); ++rank) {
    const auto& session = copied[rank];
    if (session.engine_epoch != first.engine_epoch ||
        session.worker_generation != first.worker_generation ||
        session.world_size != copied.size() || session.rank != rank ||
        session.materialization_warm_seal_root !=
            first.materialization_warm_seal_root) {
      return Status::FailedPrecondition(
          "DeepSeek rank serving sessions do not describe one generation");
    }
    auto gate = DeepSeekRankServingWorkerGate::Create(session);
    if (!gate.ok()) return gate.status();
    gates.push_back(std::move(*gate));
  }
  return DeepSeekRankServingCoordinator(std::move(copied), std::move(gates),
                                        channel);
}

DeepSeekRankServingCoordinator::DeepSeekRankServingCoordinator(
    std::vector<DeepSeekRankServingSessionBinding> sessions,
    std::vector<DeepSeekRankServingWorkerGate> gates,
    DeepSeekRankServingChannel& channel) noexcept
    : sessions_(std::move(sessions)), gates_(std::move(gates)),
      channel_(&channel), next_command_sequences_(sessions_.size(), 1) {}

Status DeepSeekRankServingCoordinator::fail(Status cause) noexcept {
  poisoned_ = true;
  active_ = false;
  const auto status = channel_->abort_generation(
      sessions_.front().engine_epoch, sessions_.front().worker_generation,
      cause);
  return status.ok() ? cause : status;
}

Status DeepSeekRankServingCoordinator::begin(
    std::uint64_t request_id, std::uint64_t request_generation,
    DeepSeekPipelinePlanDescriptor plan, std::uint64_t input_lease_identity,
    std::uint64_t output_lease_identity) {
  if (poisoned_ || active_ || request_id == 0 || request_generation == 0 ||
      plan.engine_epoch != sessions_.front().engine_epoch ||
      plan.plan_sequence == 0 || plan.token_count == 0 ||
      plan.sequence_count == 0 ||
      plan.phase == DeepSeekPlanPhase::kDrain ||
      static_cast<std::uint32_t>(plan.phase) >
          static_cast<std::uint32_t>(DeepSeekPlanPhase::kDrain) ||
      input_lease_identity == 0 ||
      output_lease_identity == 0) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving plan cannot begin");
  }
  execute_commands_.clear();
  execute_sent_.assign(sessions_.size(), false);
  completion_received_.assign(sessions_.size(), false);
  cancel_commands_.clear();
  cancel_sent_.clear();
  cancellation_requested_ = false;
  complete_ = false;
  for (std::uint32_t rank = 0; rank < sessions_.size(); ++rank) {
    if (next_command_sequences_[rank] == 0 ||
        next_command_sequences_[rank] == std::numeric_limits<std::uint64_t>::max()) {
      return fail(Status::FailedPrecondition(
          "DeepSeek rank serving command sequence is exhausted"));
    }
    DeepSeekRankServingCommand command;
    command.session = sessions_[rank];
    command.command_sequence = next_command_sequences_[rank]++;
    command.request_id = request_id;
    command.request_generation = request_generation;
    command.plan = plan;
    command.input_lease_identity = rank == 0 ? input_lease_identity : 0;
    command.output_lease_identity = rank + 1 == sessions_.size()
                                        ? output_lease_identity
                                        : 0;
    execute_commands_.push_back(command);
  }
  active_ = true;
  return Status::Ok();
}

Status DeepSeekRankServingCoordinator::cancel(
    std::uint64_t request_id, std::uint64_t request_generation) {
  if (poisoned_ || !active_ || cancellation_requested_ ||
      execute_commands_.empty() || request_id != execute_commands_[0].request_id ||
      request_generation != execute_commands_[0].request_generation) {
    return Status::FailedPrecondition("DeepSeek rank serving plan cannot cancel");
  }
  for (const bool sent : execute_sent_) {
    if (!sent) {
      return Status::Unavailable(
          "DeepSeek rank serving cannot cancel before execute delivery");
    }
  }
  cancel_commands_.clear();
  cancel_sent_.assign(sessions_.size(), false);
  for (std::uint32_t rank = 0; rank < sessions_.size(); ++rank) {
    if (next_command_sequences_[rank] == 0 ||
        next_command_sequences_[rank] == std::numeric_limits<std::uint64_t>::max()) {
      return fail(Status::FailedPrecondition(
          "DeepSeek rank serving cancellation sequence is exhausted"));
    }
    auto command = execute_commands_[rank];
    command.command_sequence = next_command_sequences_[rank]++;
    command.kind = DeepSeekRankServingCommandKind::kCancel;
    command.target_execution_sequence = execute_commands_[rank].command_sequence;
    command.input_lease_identity = 0;
    command.output_lease_identity = 0;
    cancel_commands_.push_back(command);
  }
  cancellation_requested_ = true;
  return Status::Ok();
}

Status DeepSeekRankServingCoordinator::send_pending(
    std::vector<DeepSeekRankServingCommand>& commands, std::vector<bool>& sent) {
  for (std::uint32_t rank = 0; rank < commands.size(); ++rank) {
    if (sent[rank]) continue;
    const auto frame = encode_deepseek_rank_serving_command(commands[rank]);
    const auto status = channel_->send_command(rank, frame);
    if (status.code() == StatusCode::kUnavailable) return Status::Ok();
    if (!status.ok()) return fail(status);
    const auto accepted = gates_[rank].accept(commands[rank]);
    if (!accepted.ok()) return fail(accepted);
    sent[rank] = true;
  }
  return Status::Ok();
}

Status DeepSeekRankServingCoordinator::poll_completions() {
  for (std::uint32_t rank = 0; rank < sessions_.size(); ++rank) {
    if (completion_received_[rank]) continue;
    auto frame = channel_->poll_completion(rank);
    if (!frame.ok()) return fail(frame.status());
    if (!*frame) continue;
    auto completion = decode_deepseek_rank_serving_completion(**frame);
    if (!completion.ok()) return fail(completion.status());
    const auto status = gates_[rank].complete(*completion);
    if (!status.ok()) return fail(status);
    completion_received_[rank] = true;
  }
  for (const bool received : completion_received_) {
    if (!received) return Status::Ok();
  }
  active_ = false;
  complete_ = true;
  return Status::Ok();
}

Status DeepSeekRankServingCoordinator::advance() {
  if (poisoned_) return Status::FailedPrecondition("DeepSeek rank serving is poisoned");
  if (!active_) return Status::Ok();
  auto status = send_pending(execute_commands_, execute_sent_);
  if (!status.ok()) return status;
  for (const bool sent : execute_sent_) {
    if (!sent) return Status::Ok();
  }
  if (cancellation_requested_) {
    status = send_pending(cancel_commands_, cancel_sent_);
    if (!status.ok()) return status;
    for (const bool sent : cancel_sent_) {
      if (!sent) return Status::Ok();
    }
  }
  return poll_completions();
}

}  // namespace pih
