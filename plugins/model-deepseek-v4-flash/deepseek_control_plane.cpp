#include "pih/model/deepseek_control_plane.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace pih {

Result<DeepSeekControlPlane> DeepSeekControlPlane::Create(
    std::uint64_t engine_epoch, DeepSeekPipelineCapacity capacity) {
  if (engine_epoch == 0) {
    return Status::InvalidArgument("DeepSeek control epoch must be nonzero");
  }
  auto resources = DeepSeekPipelineResourceSet::Create(std::move(capacity));
  if (!resources.ok()) return resources.status();
  auto output_credits = OutputBurstCreditPool::Create(
      {DeepSeekPipelineCapacity::kBoundaryCredits, 1, 5, 64, 1U << 20U});
  if (!output_credits.ok()) return output_credits.status();
  return DeepSeekControlPlane(
      engine_epoch, std::make_unique<DeepSeekRequestRegistry>(),
      std::make_unique<DeepSeekPipelineResourceSet>(std::move(*resources)),
      std::move(*output_credits));
}

Status DeepSeekControlPlane::submit(std::uint64_t request_id,
                                    std::uint64_t request_generation) {
  return requests_->submit(request_id, request_generation);
}

Status DeepSeekControlPlane::cancel(std::uint64_t request_id,
                                    std::uint64_t request_generation) {
  const bool active = execution_.has_value() &&
                      execution_->contains_request(request_id,
                                                   request_generation);
  if (active) {
    const auto validation =
        execution_->validate_cancel_request(request_id, request_generation);
    if (!validation.ok()) return validation;
    if (execution_->state() == DeepSeekPipelineCoordinatorState::kPreparing ||
        execution_->state() ==
            DeepSeekPipelineCoordinatorState::kReadyToCommit) {
      if (!current_output_burst_.has_value()) {
        return Status::Internal(
            "DeepSeek prepared cancellation lost output burst ownership");
      }
      const auto output =
          output_credits_.validate_abort(*current_output_burst_);
      if (!output.ok()) return output;
    }
  }
  Status status = active
                      ? execution_->cancel_request(request_id,
                                                   request_generation)
                      : requests_->cancel(request_id, request_generation);
  if (execution_.has_value() &&
      execution_->state() == DeepSeekPipelineCoordinatorState::kAborted) {
    if (current_output_burst_.has_value()) {
      const auto credit = output_credits_.abort(*current_output_burst_);
      if (!credit.ok()) return credit;
      current_output_burst_.reset();
      current_output_owners_.clear();
    }
    execution_.reset();
  }
  return status;
}

Status DeepSeekControlPlane::retire(std::uint64_t request_id,
                                    std::uint64_t request_generation) {
  const auto validation = validate_retire(request_id, request_generation);
  if (!validation.ok()) return validation;
  const auto status = requests_->retire(request_id, request_generation);
  if (status.ok()) ledgers_.erase({request_id, request_generation});
  return status;
}

Status DeepSeekControlPlane::validate_retire(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  const auto request =
      requests_->validate_retire(request_id, request_generation);
  if (!request.ok()) return request;
  for (const auto& owners : pending_output_owners_) {
    if (!owners.has_value()) continue;
    if (std::ranges::any_of(*owners, [&](const auto& identity) {
          return identity.request_id == request_id &&
                 identity.request_generation == request_generation;
        })) {
      return Status::FailedPrecondition(
          "DeepSeek request cannot retire before output acknowledgement");
    }
  }
  return Status::Ok();
}

Status DeepSeekControlPlane::finish_request(
    std::uint64_t request_id, std::uint64_t request_generation) {
  return requests_->finish(request_id, request_generation);
}

Status DeepSeekControlPlane::configure_ledger(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t prompt_token_count,
    std::uint32_t maximum_completion_tokens,
    std::uint32_t minimum_completion_tokens) {
  auto state = requests_->state(request_id, request_generation);
  if (!state.ok()) return state.status();
  if (*state != DeepSeekRequestState::kAdmitted || prompt_token_count == 0 ||
      maximum_completion_tokens == 0 ||
      minimum_completion_tokens > maximum_completion_tokens ||
      prompt_token_count >
          std::numeric_limits<std::uint64_t>::max() - maximum_completion_tokens) {
    return Status::InvalidArgument("DeepSeek accepted-token ledger is invalid");
  }
  const LedgerKey key{request_id, request_generation};
  if (ledgers_.contains(key)) {
    return Status::FailedPrecondition(
        "DeepSeek accepted-token ledger already exists");
  }
  ledgers_.emplace(key, Ledger{prompt_token_count, maximum_completion_tokens,
                               minimum_completion_tokens, {},
                               DeepSeekFinishReason::kNone});
  return Status::Ok();
}

Status DeepSeekControlPlane::configure_sampling(
    std::uint64_t request_id, std::uint64_t request_generation,
    DeepSeekRequestSamplingConfig config) {
  auto state = requests_->state(request_id, request_generation);
  if (!state.ok()) return state.status();
  auto found = ledgers_.find({request_id, request_generation});
  bool valid_stops = config.stop_token_count <=
                     DeepSeekRequestSamplingConfig::kMaximumStopTokenIds;
  for (std::uint32_t index = 0; valid_stops &&
       index < config.stop_token_ids.size(); ++index) {
    if (index >= config.stop_token_count) {
      valid_stops = config.stop_token_ids[index] == 0;
      continue;
    }
    valid_stops = config.stop_token_ids[index] < 129280U;
    for (std::uint32_t prior = 0; valid_stops && prior < index; ++prior)
      valid_stops = config.stop_token_ids[prior] != config.stop_token_ids[index];
  }
  const bool valid_mode =
      ((config.mode == DeepSeekSamplingMode::kGreedy &&
        config.temperature == 0.0F && config.top_p == 1.0F &&
        !config.top_k.has_value()) ||
       (config.mode == DeepSeekSamplingMode::kStochastic &&
        std::isfinite(config.temperature) && config.temperature > 0.0F &&
        config.temperature <= 2.0F && std::isfinite(config.top_p) &&
        config.top_p > 0.0F && config.top_p <= 1.0F &&
        (!config.top_k.has_value() ||
         (*config.top_k > 0 && *config.top_k <= 129280U)))) &&
      config.top_logprobs_count <= 20U &&
      (config.logprobs_enabled || config.top_logprobs_count == 0U) &&
      valid_stops;
  if (*state != DeepSeekRequestState::kAdmitted ||
      found == ledgers_.end() || config.config_id == 0 || !valid_mode ||
      found->second.sampling.has_value() || !found->second.token_ids.empty()) {
    return Status::FailedPrecondition(
        "DeepSeek request sampling configuration is not installable");
  }
  found->second.sampling = config;
  return Status::Ok();
}

Result<DeepSeekRequestSamplingConfig>
DeepSeekControlPlane::sampling_config(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  const auto found = ledgers_.find({request_id, request_generation});
  if (found == ledgers_.end() || !found->second.sampling.has_value()) {
    return Status::FailedPrecondition(
        "DeepSeek request sampling configuration is unavailable");
  }
  return *found->second.sampling;
}

Status DeepSeekControlPlane::validate_batch(
    const Ledger& ledger, const DeepSeekAcceptedTokenBatch& batch) const {
  if (ledger.finish_reason != DeepSeekFinishReason::kNone ||
      batch.token_ids.empty() ||
      batch.token_ids.size() >
          ledger.maximum_completion_tokens - ledger.token_ids.size()) {
    return Status::FailedPrecondition(
        "DeepSeek accepted-token batch exceeds live ledger");
  }
  for (const auto token : batch.token_ids) {
    if (token >= 129280U) {
      return Status::InvalidArgument(
          "DeepSeek accepted token is outside vocabulary");
    }
  }
  if (ledger.sampling.has_value()) {
    const auto& config = *ledger.sampling;
    if (!batch.selected_logprobs.empty() &&
        batch.selected_logprobs.size() != batch.token_ids.size()) {
      return Status::InvalidArgument(
          "DeepSeek selected logprob cardinality differs");
    }
    for (std::size_t index = 0; index < batch.selected_logprobs.size();
         ++index) {
      const auto& receipt = batch.selected_logprobs[index];
      if (receipt.config_id != config.config_id ||
          !std::isfinite(receipt.selected_logprob)) {
        return Status::InvalidArgument(
            "DeepSeek selected logprob receipt is stale or invalid");
      }
      if (receipt.top_logprobs.size() != config.top_logprobs_count) {
        return Status::InvalidArgument(
            "DeepSeek top logprob cardinality differs");
      }
      for (std::size_t top_index = 0;
           top_index < receipt.top_logprobs.size(); ++top_index) {
        const auto& candidate = receipt.top_logprobs[top_index];
        if (candidate.token_id >= 129280U ||
            !std::isfinite(candidate.logprob) ||
            candidate.rank != top_index + 1U ||
            (top_index != 0U &&
             receipt.top_logprobs[top_index - 1U].logprob <
                 candidate.logprob)) {
          return Status::InvalidArgument(
              "DeepSeek ranked top logprob receipt is invalid");
        }
        if (candidate.token_id == batch.token_ids[index] &&
            candidate.logprob != receipt.selected_logprob) {
          return Status::InvalidArgument(
              "DeepSeek selected and ranked logprobs disagree");
        }
        for (std::size_t prior = 0; prior < top_index; ++prior) {
          if (receipt.top_logprobs[prior].token_id == candidate.token_id) {
            return Status::InvalidArgument(
                "DeepSeek ranked top logprob tokens are duplicated");
          }
        }
      }
    }
    if (config.mode == DeepSeekSamplingMode::kGreedy) {
      if (!batch.sampling_receipts.empty()) {
        return Status::InvalidArgument(
            "DeepSeek greedy result must not consume RNG receipts");
      }
    } else {
      if (batch.sampling_receipts.size() != batch.token_ids.size()) {
        return Status::InvalidArgument(
            "DeepSeek stochastic result receipt cardinality differs");
      }
      for (std::size_t index = 0; index < batch.sampling_receipts.size();
           ++index) {
        const auto& receipt = batch.sampling_receipts[index];
        if (receipt.config_id != config.config_id ||
            receipt.sample_ordinal != ledger.sample_ordinal + index ||
            !std::isfinite(receipt.selected_logprob)) {
          return Status::InvalidArgument(
              "DeepSeek stochastic sampling receipt is stale or invalid");
        }
        if (!batch.selected_logprobs.empty() &&
            batch.selected_logprobs[index].selected_logprob !=
                receipt.selected_logprob) {
          return Status::InvalidArgument(
              "DeepSeek sampling and logprob receipts disagree");
        }
      }
    }
  } else if (!batch.sampling_receipts.empty() ||
             !batch.selected_logprobs.empty()) {
    return Status::InvalidArgument(
        "DeepSeek sampling receipt has no request configuration");
  }
  const auto target = ledger.token_ids.size() + batch.token_ids.size();
  if ((batch.finish_reason == DeepSeekFinishReason::kStop ||
       batch.finish_reason == DeepSeekFinishReason::kToolCalls) &&
      target < ledger.minimum_completion_tokens) {
    return Status::InvalidArgument("DeepSeek terminal violates min_tokens");
  }
  if (batch.finish_reason == DeepSeekFinishReason::kLength &&
      target != ledger.maximum_completion_tokens) {
    return Status::InvalidArgument(
        "DeepSeek length finish differs from max_tokens");
  }
  return Status::Ok();
}

Status DeepSeekControlPlane::stage_accepted_tokens(
    std::uint64_t plan_sequence,
    std::vector<DeepSeekAcceptedTokenBatch> batches) {
  if (!execution_.has_value() ||
      execution_->state() != DeepSeekPipelineCoordinatorState::kCommitted ||
      execution_->plan_sequence() != plan_sequence ||
      tentative_accepted_.has_value() ||
      batches.size() != execution_->request_identities().size()) {
    return Status::FailedPrecondition(
        "DeepSeek accepted-token result does not match committed plan");
  }
  const auto& identities = execution_->request_identities();
  for (std::size_t index = 0; index < identities.size(); ++index) {
    const auto& identity = identities[index];
    const auto ledger = ledgers_.find(
        {identity.request_id, identity.request_generation});
    if (ledger == ledgers_.end()) {
      return Status::FailedPrecondition(
          "DeepSeek accepted-token ledger is missing");
    }
    const auto status = validate_batch(ledger->second, batches[index]);
    if (!status.ok()) return status;
  }
  tentative_accepted_.emplace(
      TentativeAccepted{plan_sequence, std::move(batches)});
  return Status::Ok();
}

Status DeepSeekControlPlane::prepare_plan(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRequestIdentity> identities) {
  if (execution_.has_value() || current_output_burst_.has_value() ||
      !current_output_owners_.empty()) {
    return Status::ResourceExhausted(
        "DeepSeek control plane already owns an active execution");
  }
  if (descriptor.engine_epoch != engine_epoch_) {
    return Status::FailedPrecondition(
        "DeepSeek plan epoch differs from control plane");
  }
  auto output_owners = identities;
  auto output_burst = output_credits_.acquire(
      descriptor.plan_sequence, OutputPlanKind::kDeepSeek, 5, 64, 1U << 20U);
  if (!output_burst.ok()) return output_burst.status();
  auto execution = DeepSeekPipelineExecution::Create(
      *requests_, *resources_, descriptor, std::move(identities));
  if (!execution.ok()) {
    const auto rollback = output_credits_.abort(*output_burst);
    if (!rollback.ok()) {
      return Status::Internal("DeepSeek output credit rollback failed");
    }
    return execution.status();
  }
  current_output_burst_.emplace(*output_burst);
  current_output_owners_ = std::move(output_owners);
  execution_.emplace(std::move(*execution));
  return Status::Ok();
}

Status DeepSeekControlPlane::stage_ready(std::uint32_t rank) {
  if (!execution_.has_value()) {
    return Status::FailedPrecondition("DeepSeek has no active execution");
  }
  return execution_->stage_ready(rank);
}

Status DeepSeekControlPlane::stage_reject(std::uint32_t rank, Status reason) {
  if (!execution_.has_value() || !current_output_burst_.has_value()) {
    return Status::FailedPrecondition("DeepSeek has no active execution");
  }
  const auto execution = execution_->validate_stage_reject(rank);
  if (!execution.ok()) return execution;
  const auto output = output_credits_.validate_abort(*current_output_burst_);
  if (!output.ok()) return output;
  const auto status = execution_->stage_reject(rank, std::move(reason));
  if (execution_->state() == DeepSeekPipelineCoordinatorState::kAborted) {
    if (current_output_burst_.has_value()) {
      const auto credit = output_credits_.abort(*current_output_burst_);
      if (!credit.ok()) return credit;
      current_output_burst_.reset();
      current_output_owners_.clear();
    }
    tentative_accepted_.reset();
    execution_.reset();
  }
  return status;
}

Status DeepSeekControlPlane::validate_commit_plan() const {
  if (!execution_.has_value() || !current_output_burst_.has_value()) {
    return Status::FailedPrecondition("DeepSeek has no active execution");
  }
  auto credit = output_credits_.validate_commit(*current_output_burst_);
  if (!credit.ok()) return credit;
  return execution_->validate_commit();
}

Status DeepSeekControlPlane::commit_plan() {
  const auto validation = validate_commit_plan();
  if (!validation.ok()) return validation;
  auto credit = output_credits_.commit(*current_output_burst_);
  if (!credit.ok()) return credit;
  return execution_->commit();
}

Status DeepSeekControlPlane::prepare_stage_complete(std::uint32_t rank) {
  if (!execution_.has_value() || !current_output_burst_.has_value()) {
    return Status::FailedPrecondition("DeepSeek has no active execution");
  }
  auto pending = std::find_if(
      pending_output_bursts_.begin(), pending_output_bursts_.end(),
      [](const auto& lease) { return !lease.has_value(); });
  if (pending == pending_output_bursts_.end()) {
    return Status::Internal("DeepSeek output burst has no pending owner slot");
  }
  const auto pending_index = static_cast<std::size_t>(
      pending - pending_output_bursts_.begin());
  if (pending_output_owners_[pending_index].has_value()) {
    return Status::Internal("DeepSeek output owner slot is already occupied");
  }
  const auto& execution_owners = execution_->request_identities();
  if (current_output_owners_.size() != execution_owners.size()) {
    return Status::Internal("DeepSeek current output ownership drifted");
  }
  for (std::size_t index = 0; index < execution_owners.size(); ++index) {
    if (current_output_owners_[index].request_id !=
            execution_owners[index].request_id ||
        current_output_owners_[index].request_generation !=
            execution_owners[index].request_generation) {
      return Status::Internal("DeepSeek current output identity drifted");
    }
  }
  auto transfer = output_credits_.validate_transfer(*current_output_burst_);
  if (!transfer.ok()) return transfer;
  if (tentative_accepted_.has_value()) {
    const auto& identities = execution_->request_identities();
    if (tentative_accepted_->batches.size() != identities.size()) {
      return Status::Internal(
          "DeepSeek accepted-token staging cardinality drifted");
    }
    for (std::size_t index = 0; index < identities.size(); ++index) {
      const auto& identity = identities[index];
      auto state = requests_->state(identity.request_id,
                                    identity.request_generation);
      if (!state.ok()) return state.status();
      if (*state != DeepSeekRequestState::kCommitted) continue;
      auto ledger = ledgers_.find(
          {identity.request_id, identity.request_generation});
      if (ledger == ledgers_.end()) {
        return Status::Internal(
            "DeepSeek accepted-token ledger disappeared before completion");
      }
      const auto& batch = tentative_accepted_->batches[index];
      ledger->second.token_ids.reserve(
          ledger->second.token_ids.size() + batch.token_ids.size());
      ledger->second.selected_logprobs.reserve(
          ledger->second.selected_logprobs.size() +
          batch.selected_logprobs.size());
      ledger->second.top_logprobs.reserve(
          ledger->second.top_logprobs.size() +
          batch.selected_logprobs.size());
    }
  }
  return execution_->validate_stage_complete(rank);
}

Status DeepSeekControlPlane::stage_complete(std::uint32_t rank) {
  const auto prepared = prepare_stage_complete(rank);
  if (!prepared.ok()) return prepared;
  auto pending = std::find_if(
      pending_output_bursts_.begin(), pending_output_bursts_.end(),
      [](const auto& lease) { return !lease.has_value(); });
  const auto pending_index = static_cast<std::size_t>(
      pending - pending_output_bursts_.begin());
  const auto status = execution_->stage_complete(rank);
  if (status.ok() &&
      execution_->state() == DeepSeekPipelineCoordinatorState::kComplete) {
    auto transferred = output_credits_.transfer(*current_output_burst_);
    if (!transferred.ok()) return transferred;
    if (tentative_accepted_.has_value()) {
      const auto& identities = execution_->request_identities();
      for (std::size_t index = 0; index < identities.size(); ++index) {
        const auto& identity = identities[index];
        auto state = requests_->state(identity.request_id,
                                      identity.request_generation);
        if (!state.ok() || (*state != DeepSeekRequestState::kAdmitted &&
                            *state != DeepSeekRequestState::kCompleted)) {
          continue;
        }
        auto& ledger = ledgers_.at(
            {identity.request_id, identity.request_generation});
        auto& batch = tentative_accepted_->batches[index];
        ledger.token_ids.insert(ledger.token_ids.end(), batch.token_ids.begin(),
                                batch.token_ids.end());
        ledger.finish_reason =
            batch.finish_reason == DeepSeekFinishReason::kNone &&
                    ledger.token_ids.size() == ledger.maximum_completion_tokens
                ? DeepSeekFinishReason::kLength
                : batch.finish_reason;
        ledger.sample_ordinal += batch.sampling_receipts.size();
        for (auto& receipt : batch.selected_logprobs) {
          ledger.selected_logprobs.push_back(receipt.selected_logprob);
          ledger.top_logprobs.push_back(std::move(receipt.top_logprobs));
        }
      }
    }
    tentative_accepted_.reset();
    pending->emplace(*current_output_burst_);
    pending_output_owners_[pending_index].emplace(
        std::move(current_output_owners_));
    current_output_burst_.reset();
    execution_.reset();
  }
  return status;
}

Status DeepSeekControlPlane::acknowledge_output_plan(
    std::uint64_t plan_sequence) {
  auto pending = std::find_if(
      pending_output_bursts_.begin(), pending_output_bursts_.end(),
      [plan_sequence](const auto& lease) {
        return lease.has_value() && lease->plan_sequence == plan_sequence;
      });
  if (pending == pending_output_bursts_.end()) {
    return Status::FailedPrecondition("DeepSeek output ACK is stale or forged");
  }
  const auto pending_index = static_cast<std::size_t>(
      pending - pending_output_bursts_.begin());
  if (!pending_output_owners_[pending_index].has_value()) {
    return Status::Internal("DeepSeek output ACK lost request ownership");
  }
  auto status = output_credits_.release(**pending);
  if (status.ok()) {
    pending->reset();
    pending_output_owners_[pending_index].reset();
  }
  return status;
}

Status DeepSeekControlPlane::stage_failed(std::uint32_t rank, Status reason) {
  if (!execution_.has_value() || !current_output_burst_.has_value()) {
    return Status::FailedPrecondition("DeepSeek has no active execution");
  }
  const auto execution = execution_->validate_stage_failed(rank);
  if (!execution.ok()) return execution;
  auto pending = std::find_if(
      pending_output_bursts_.begin(), pending_output_bursts_.end(),
      [](const auto& lease) { return !lease.has_value(); });
  if (pending == pending_output_bursts_.end()) {
    return Status::Internal(
        "DeepSeek failed output burst has no pending owner slot");
  }
  const auto pending_index = static_cast<std::size_t>(
      pending - pending_output_bursts_.begin());
  if (pending_output_owners_[pending_index].has_value()) {
    return Status::Internal("DeepSeek failed output owner slot is occupied");
  }
  const auto& execution_owners = execution_->request_identities();
  if (current_output_owners_.size() != execution_owners.size()) {
    return Status::Internal("DeepSeek failed output ownership drifted");
  }
  for (std::size_t index = 0; index < execution_owners.size(); ++index) {
    if (current_output_owners_[index].request_id !=
            execution_owners[index].request_id ||
        current_output_owners_[index].request_generation !=
            execution_owners[index].request_generation) {
      return Status::Internal("DeepSeek failed output identity drifted");
    }
  }
  auto transfer = output_credits_.validate_transfer(*current_output_burst_);
  if (!transfer.ok()) return transfer;
  const auto status = execution_->stage_failed(rank, std::move(reason));
  if (execution_->state() == DeepSeekPipelineCoordinatorState::kPoisoned) {
    auto transferred = output_credits_.transfer(*current_output_burst_);
    if (!transferred.ok()) return transferred;
    pending->emplace(*current_output_burst_);
    pending_output_owners_[pending_index].emplace(
        std::move(current_output_owners_));
    current_output_burst_.reset();
    tentative_accepted_.reset();
    requests_->fail_all();
  }
  return status;
}

void DeepSeekControlPlane::fail_all() noexcept { requests_->fail_all(); }

void DeepSeekControlPlane::cancel_all() noexcept { requests_->cancel_all(); }

Result<DeepSeekRequestState> DeepSeekControlPlane::request_state(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  return requests_->state(request_id, request_generation);
}

Result<DeepSeekPipelineCoordinatorState>
DeepSeekControlPlane::execution_state() const {
  if (!execution_.has_value()) {
    return Status::FailedPrecondition("DeepSeek has no active execution");
  }
  return execution_->state();
}

Result<DeepSeekAcceptedTokenSnapshot>
DeepSeekControlPlane::accepted_token_snapshot(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  const auto found = ledgers_.find({request_id, request_generation});
  if (found == ledgers_.end()) {
    return Status::FailedPrecondition(
        "DeepSeek accepted-token ledger is missing");
  }
  const auto& ledger = found->second;
  const auto accepted = static_cast<std::uint64_t>(ledger.token_ids.size());
  const auto processed = accepted == 0
                             ? ledger.prompt_token_count
                             : ledger.prompt_token_count + accepted - 1;
  std::optional<std::uint32_t> pending;
  if (accepted != 0 && ledger.finish_reason == DeepSeekFinishReason::kNone) {
    pending = ledger.token_ids.back();
  }
  return DeepSeekAcceptedTokenSnapshot{ledger.token_ids, accepted, processed,
                                       pending, ledger.finish_reason,
                                       ledger.sample_ordinal,
                                       ledger.selected_logprobs,
                                       ledger.top_logprobs,
                                       ledger.minimum_completion_tokens};
}

}  // namespace pih
