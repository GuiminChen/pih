#include "pih/model/deepseek_nccl_p2p_plan.h"

#include <limits>

namespace pih {
namespace {

Status validate_manifest(const DeepSeekNcclP2pManifest& m) {
  if (m.operation_plan_id == 0 || m.engine_epoch == 0 ||
      m.communicator_generation == 0 || m.pipeline_plan_sequence == 0 ||
      m.operation_ordinal == 0 || m.global_issue_ordinal == 0 ||
      m.buffer_owner_id == 0 ||
      m.buffer_generation == 0 || m.context_identity == 0 ||
      m.token_count == 0 ||
      m.communicator_local_rank > 1 || m.communicator_peer_rank > 1 ||
      m.communicator_local_rank == m.communicator_peer_rank ||
      static_cast<std::uint8_t>(m.role) >
          static_cast<std::uint8_t>(DeepSeekNcclRole::kRecv)) {
    return Status::InvalidArgument("DeepSeek NCCL manifest identity is invalid");
  }
  const std::uint64_t bytes = std::uint64_t{m.token_count} * 32768U;
  if (m.buffer_offset_bytes > m.buffer_capacity_bytes ||
      bytes > m.buffer_capacity_bytes - m.buffer_offset_bytes ||
      std::uint64_t{m.token_count} * 4U * 4096U >
          std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("DeepSeek NCCL buffer range is invalid");
  }
  const bool send = m.role == DeepSeekNcclRole::kSend;
  if ((send && (m.local_global_rank != m.directed_boundary_id ||
                m.peer_global_rank != m.directed_boundary_id + 1 ||
                m.communicator_local_rank != 0 || m.communicator_peer_rank != 1)) ||
      (!send && (m.local_global_rank != m.directed_boundary_id + 1 ||
                 m.peer_global_rank != m.directed_boundary_id ||
                 m.communicator_local_rank != 1 || m.communicator_peer_rank != 0))) {
    return Status::InvalidArgument("DeepSeek NCCL directed rank mapping is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekNcclP2pPlan> DeepSeekNcclP2pPlan::Create(
    DeepSeekNcclP2pManifest manifest) {
  const auto valid = validate_manifest(manifest);
  if (!valid.ok()) return valid;
  return DeepSeekNcclP2pPlan(manifest);
}

Status DeepSeekNcclP2pPlan::ValidatePair(
    const DeepSeekNcclP2pManifest& send,
    const DeepSeekNcclP2pManifest& recv) {
  const auto send_valid = validate_manifest(send);
  if (!send_valid.ok()) return send_valid;
  const auto recv_valid = validate_manifest(recv);
  if (!recv_valid.ok()) return recv_valid;
  if (send.role != DeepSeekNcclRole::kSend ||
      recv.role != DeepSeekNcclRole::kRecv ||
      send.operation_plan_id != recv.operation_plan_id ||
      send.engine_epoch != recv.engine_epoch ||
      send.communicator_generation != recv.communicator_generation ||
      send.pipeline_plan_sequence != recv.pipeline_plan_sequence ||
      send.global_issue_ordinal != recv.global_issue_ordinal ||
      send.directed_boundary_id != recv.directed_boundary_id ||
      send.token_count != recv.token_count ||
      send.local_global_rank != recv.peer_global_rank ||
      send.peer_global_rank != recv.local_global_rank) {
    return Status::InvalidArgument("DeepSeek NCCL send/recv manifests do not match");
  }
  return Status::Ok();
}

Status DeepSeekNcclP2pPlan::poison(Status status) {
  state_ = DeepSeekNcclP2pState::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek NCCL plan poisoned") : status;
}

Status DeepSeekNcclP2pPlan::issue(DeepSeekNcclP2pDriver& driver) {
  if (state_ != DeepSeekNcclP2pState::kPlanned) {
    return Status::FailedPrecondition("DeepSeek NCCL plan is not issuable");
  }
  auto status = driver.group_start();
  if (!status.ok()) return poison(status);
  status = manifest_.role == DeepSeekNcclRole::kSend
               ? driver.send(element_count(), manifest_.communicator_peer_rank)
               : driver.recv(element_count(), manifest_.communicator_peer_rank);
  if (!status.ok()) return poison(status);
  auto ended = driver.group_end();
  if (!ended.ok()) return poison(ended.status());
  if (*ended == DeepSeekNcclAsyncStatus::kError) {
    return poison(Status::Internal("DeepSeek NCCL group end failed"));
  }
  state_ = *ended == DeepSeekNcclAsyncStatus::kSuccess
               ? DeepSeekNcclP2pState::kIssuedToStream
               : DeepSeekNcclP2pState::kGroupEndPending;
  return Status::Ok();
}

Status DeepSeekNcclP2pPlan::poll_issue(DeepSeekNcclP2pDriver& driver) {
  if (state_ != DeepSeekNcclP2pState::kGroupEndPending) {
    return Status::FailedPrecondition("DeepSeek NCCL issue is not pending");
  }
  auto status = driver.async_status();
  if (!status.ok()) return poison(status.status());
  if (*status == DeepSeekNcclAsyncStatus::kError) {
    return poison(Status::Internal("DeepSeek NCCL async issue failed"));
  }
  if (*status == DeepSeekNcclAsyncStatus::kSuccess) {
    state_ = DeepSeekNcclP2pState::kIssuedToStream;
  }
  return Status::Ok();
}

Status DeepSeekNcclP2pPlan::record_completion(
    CompletionEventSlot& event, CompletionEventDriver& driver,
    DriverStreamHandle stream) {
  if (state_ != DeepSeekNcclP2pState::kIssuedToStream) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL operation was not issued before CUDA event");
  }
  const auto status = event.record(driver, stream, manifest_.operation_ordinal);
  if (!status.ok()) return poison(status);
  state_ = DeepSeekNcclP2pState::kDeviceInFlight;
  return Status::Ok();
}

Status DeepSeekNcclP2pPlan::poll_completion(
    DeepSeekNcclP2pDriver& nccl, CompletionEventSlot& event,
    CompletionEventDriver& event_driver, CudaCompletionFrontier& frontier,
    CompletionEvidenceProvider& evidence) {
  if (state_ != DeepSeekNcclP2pState::kDeviceInFlight) {
    return Status::FailedPrecondition("DeepSeek NCCL operation is not in flight");
  }
  if (event.state() == CompletionEventSlotState::kRecorded) {
    const auto completed = event.poll(event_driver, frontier, evidence);
    if (!completed.ok()) {
      if (completed.code() == StatusCode::kUnavailable && !frontier.poisoned()) {
        return completed;
      }
      return poison(completed);
    }
  }
  auto async = nccl.async_status();
  if (!async.ok()) return poison(async.status());
  if (*async == DeepSeekNcclAsyncStatus::kError) {
    return poison(Status::Internal("DeepSeek NCCL completion async check failed"));
  }
  if (*async == DeepSeekNcclAsyncStatus::kInProgress) {
    return Status::Unavailable("DeepSeek NCCL completion is not closed");
  }
  state_ = DeepSeekNcclP2pState::kCompleteVerified;
  return Status::Ok();
}

Result<DeepSeekNcclOperationSequencer>
DeepSeekNcclOperationSequencer::Create(
    std::uint64_t first_operation_ordinal) {
  if (first_operation_ordinal == 0) {
    return Status::InvalidArgument("DeepSeek NCCL first ordinal is invalid");
  }
  return DeepSeekNcclOperationSequencer(first_operation_ordinal);
}

Status DeepSeekNcclOperationSequencer::issue(
    DeepSeekNcclP2pPlan& plan, DeepSeekNcclP2pDriver& driver) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek NCCL sequencer is poisoned");
  }
  if (active_plan_id_ != 0 ||
      plan.manifest().operation_ordinal != next_ordinal_) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL operation violates rank-local order");
  }
  active_plan_id_ = plan.manifest().operation_plan_id;
  const auto status = plan.issue(driver);
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  return Status::Ok();
}

Status DeepSeekNcclOperationSequencer::close(
    const DeepSeekNcclP2pPlan& plan) {
  if (poisoned_) {
    return Status::Unavailable("DeepSeek NCCL sequencer is poisoned");
  }
  if (active_plan_id_ == 0 ||
      active_plan_id_ != plan.manifest().operation_plan_id ||
      plan.manifest().operation_ordinal != next_ordinal_ ||
      plan.state() != DeepSeekNcclP2pState::kCompleteVerified) {
    if (plan.state() == DeepSeekNcclP2pState::kPoisoned) poisoned_ = true;
    return Status::FailedPrecondition(
        "DeepSeek NCCL operation cannot close sequencer frontier");
  }
  if (next_ordinal_ == std::numeric_limits<std::uint64_t>::max()) {
    poisoned_ = true;
    return Status::ResourceExhausted("DeepSeek NCCL ordinal exhausted");
  }
  ++next_ordinal_;
  active_plan_id_ = 0;
  return Status::Ok();
}

}  // namespace pih
