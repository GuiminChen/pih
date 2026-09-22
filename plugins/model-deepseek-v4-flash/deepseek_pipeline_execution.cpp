#include "pih/model/deepseek_pipeline_execution.h"

#include <new>
#include <unordered_set>
#include <utility>

namespace pih {

DeepSeekPipelineExecution& DeepSeekPipelineExecution::operator=(
    DeepSeekPipelineExecution&& other) noexcept {
  if (this != &other) {
    this->~DeepSeekPipelineExecution();
    ::new (static_cast<void*>(this))
        DeepSeekPipelineExecution(std::move(other));
  }
  return *this;
}

Result<DeepSeekPipelineExecution> DeepSeekPipelineExecution::Create(
    DeepSeekRequestRegistry& requests,
    DeepSeekPipelineResourceSet& resources,
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRequestIdentity> request_identities) {
  if (request_identities.size() != descriptor.sequence_count) {
    return Status::InvalidArgument(
        "DeepSeek pipeline request count differs from descriptor");
  }
  std::unordered_set<std::uint64_t> unique_ids;
  for (const auto& identity : request_identities) {
    if (identity.request_id == 0 || identity.request_generation == 0 ||
        !unique_ids.insert(identity.request_id).second) {
      return Status::InvalidArgument(
          "DeepSeek pipeline request identities are invalid");
    }
  }
  for (const auto& identity : request_identities) {
    const auto status = requests.validate_prepare(
        identity.request_id, identity.request_generation,
        descriptor.plan_sequence);
    if (!status.ok()) return status;
  }
  auto prepared = resources.prepare(descriptor);
  if (!prepared.ok()) return prepared.status();
  auto transaction = std::make_unique<DeepSeekPipelineTransaction>(
      std::move(*prepared));
  auto coordinator_result = DeepSeekPipelineCoordinator::Create(
      *transaction, resources.capacity().world_size);
  if (!coordinator_result.ok()) return coordinator_result.status();
  auto coordinator = std::make_unique<DeepSeekPipelineCoordinator>(
      std::move(*coordinator_result));
  std::size_t request_prepare_count = 0;
  for (const auto& identity : request_identities) {
    const auto status = requests.prepare(identity.request_id,
                                         identity.request_generation,
                                         descriptor.plan_sequence);
    if (!status.ok()) {
      for (std::size_t index = 0; index < request_prepare_count; ++index) {
        const auto& prior = request_identities[index];
        (void)requests.abort_prepare(prior.request_id,
                                     prior.request_generation,
                                     descriptor.plan_sequence);
      }
      return status;
    }
    ++request_prepare_count;
  }
  return DeepSeekPipelineExecution(requests, std::move(request_identities),
                                   std::move(transaction),
                                   std::move(coordinator));
}

Status DeepSeekPipelineExecution::stage_ready(std::uint32_t rank) {
  return coordinator_->stage_ready(rank);
}

Status DeepSeekPipelineExecution::validate_stage_reject(
    std::uint32_t rank) const {
  auto status = coordinator_->validate_stage_reject(rank);
  if (!status.ok()) return status;
  const auto plan_sequence = transaction_->descriptor().plan_sequence;
  for (const auto& identity : request_identities_) {
    status = requests_->validate_abort_prepare(
        identity.request_id, identity.request_generation, plan_sequence);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::stage_reject(std::uint32_t rank,
                                                Status reason) {
  const auto validation = validate_stage_reject(rank);
  if (!validation.ok()) return validation;
  const auto status = coordinator_->stage_reject(rank, std::move(reason));
  if (coordinator_->state() == DeepSeekPipelineCoordinatorState::kAborted) {
    rollback_prepared_requests();
  }
  return status;
}

Status DeepSeekPipelineExecution::validate_commit() const {
  auto status = coordinator_->validate_commit();
  if (!status.ok()) return status;
  const auto plan_sequence = transaction_->descriptor().plan_sequence;
  for (const auto& identity : request_identities_) {
    status = requests_->validate_commit(
        identity.request_id, identity.request_generation, plan_sequence);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::commit() {
  auto status = validate_commit();
  if (!status.ok()) return status;
  status = coordinator_->commit();
  if (!status.ok()) return status;
  for (const auto& identity : request_identities_) {
    const auto request_status = requests_->commit(
        identity.request_id, identity.request_generation,
        transaction_->descriptor().plan_sequence);
    if (!request_status.ok()) return request_status;
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::validate_cancel_request(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  auto status = requests_->validate_cancel(request_id, request_generation);
  if (!status.ok()) return status;
  if (coordinator_->state() == DeepSeekPipelineCoordinatorState::kPreparing ||
      coordinator_->state() == DeepSeekPipelineCoordinatorState::kReadyToCommit) {
    return validate_stage_reject(0);
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::cancel_request(
    std::uint64_t request_id, std::uint64_t request_generation) {
  auto status = validate_cancel_request(request_id, request_generation);
  if (!status.ok()) return status;
  status = requests_->cancel(request_id, request_generation);
  if (!status.ok()) return status;
  if (coordinator_->state() == DeepSeekPipelineCoordinatorState::kPreparing ||
      coordinator_->state() == DeepSeekPipelineCoordinatorState::kReadyToCommit) {
    status = coordinator_->stage_reject(
        0, Status::Unavailable("DeepSeek request cancelled before COMMIT"));
    if (!status.ok() &&
        coordinator_->state() != DeepSeekPipelineCoordinatorState::kAborted) {
      return status;
    }
    rollback_prepared_requests();
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::validate_stage_complete(
    std::uint32_t rank) const {
  auto status = coordinator_->validate_stage_complete(rank);
  if (!status.ok()) return status;
  const auto plan_sequence = transaction_->descriptor().plan_sequence;
  for (const auto& identity : request_identities_) {
    status = requests_->validate_backend_complete(
        identity.request_id, identity.request_generation, plan_sequence);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::stage_complete(std::uint32_t rank) {
  auto status = validate_stage_complete(rank);
  if (!status.ok()) return status;
  status = coordinator_->stage_complete(rank);
  if (!status.ok()) return status;
  if (coordinator_->state() != DeepSeekPipelineCoordinatorState::kComplete) {
    return Status::Ok();
  }
  for (const auto& identity : request_identities_) {
    const auto request_status = requests_->backend_complete(
        identity.request_id, identity.request_generation,
        transaction_->descriptor().plan_sequence,
        identity.terminal_on_complete);
    if (!request_status.ok()) return request_status;
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::validate_stage_failed(
    std::uint32_t rank) const {
  auto status = coordinator_->validate_stage_failed(rank);
  if (!status.ok()) return status;
  for (const auto& identity : request_identities_) {
    auto current = requests_->state(identity.request_id,
                                    identity.request_generation);
    if (!current.ok()) return current.status();
  }
  return Status::Ok();
}

Status DeepSeekPipelineExecution::stage_failed(std::uint32_t rank,
                                                Status reason) {
  auto status = validate_stage_failed(rank);
  if (!status.ok()) return status;
  status = coordinator_->stage_failed(rank, std::move(reason));
  if (!status.ok() &&
      coordinator_->state() != DeepSeekPipelineCoordinatorState::kPoisoned) {
    return status;
  }
  for (const auto& identity : request_identities_) {
    auto current = requests_->state(identity.request_id,
                                    identity.request_generation);
    if (current.ok() && *current != DeepSeekRequestState::kCompleted &&
        *current != DeepSeekRequestState::kCancelled &&
        *current != DeepSeekRequestState::kFailed) {
      // Engine-level fail-stop retains committed ownership until teardown.
      (void)requests_->fail(identity.request_id,
                            identity.request_generation);
    }
  }
  return status;
}

bool DeepSeekPipelineExecution::contains_request(
    std::uint64_t request_id, std::uint64_t request_generation) const noexcept {
  for (const auto& identity : request_identities_) {
    if (identity.request_id == request_id &&
        identity.request_generation == request_generation) {
      return true;
    }
  }
  return false;
}

void DeepSeekPipelineExecution::rollback_prepared_requests() noexcept {
  const auto plan_sequence = transaction_->descriptor().plan_sequence;
  for (const auto& identity : request_identities_) {
    auto current = requests_->state(identity.request_id,
                                    identity.request_generation);
    if (current.ok() && *current == DeepSeekRequestState::kPrepared) {
      (void)requests_->abort_prepare(identity.request_id,
                                     identity.request_generation,
                                     plan_sequence);
    }
  }
}

}  // namespace pih
