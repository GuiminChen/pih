#include "pih/model/deepseek_nccl_endpoint_warmup_runner.h"

#include <limits>

namespace pih {

Result<DeepSeekNcclEndpointWarmupRunner>
DeepSeekNcclEndpointWarmupRunner::Create(
    DeepSeekNcclEndpointWarmupRunnerConfig config,
    DeepSeekBoundaryTransportDriver& transport,
    DeepSeekNcclWarmupPayloadOperations& payload) {
  const auto required = std::uint64_t{config.maximum_token_count} * 32768U;
  const auto& m = config.endpoint;
  if (m.engine_epoch == 0 || m.communicator_generation == 0 ||
      m.bootstrap_lease_id == 0 || m.bootstrap_commitment_id == 0 ||
      m.local_global_rank == m.peer_global_rank || m.communicator_local_rank > 1 ||
      config.maximum_token_count == 0 || config.first_operation_ordinal == 0 ||
      config.buffer_owner_id == 0 || config.buffer_generation == 0 ||
      config.device_buffer == nullptr || config.stream == 0 ||
      config.completion_events[0] == 0 || config.completion_events[1] == 0 ||
      config.submit_ns >= config.deadline_ns || required > config.buffer_capacity_bytes ||
      config.first_operation_ordinal == std::numeric_limits<std::uint64_t>::max()) {
    return Status::InvalidArgument("DeepSeek NCCL endpoint warm-up runner config is invalid");
  }
  auto sequencer = DeepSeekNcclOperationSequencer::Create(
      config.first_operation_ordinal);
  if (!sequencer.ok()) return sequencer.status();
  return DeepSeekNcclEndpointWarmupRunner(
      std::move(config), transport, payload, std::move(*sequencer));
}

DeepSeekNcclEndpointWarmupRunner::DeepSeekNcclEndpointWarmupRunner(
    DeepSeekNcclEndpointWarmupRunnerConfig config,
    DeepSeekBoundaryTransportDriver& transport,
    DeepSeekNcclWarmupPayloadOperations& payload,
    DeepSeekNcclOperationSequencer sequencer) noexcept
    : config_(std::move(config)), transport_(&transport), payload_(&payload),
      sequencer_(std::move(sequencer)) {}

Status DeepSeekNcclEndpointWarmupRunner::fail(Status status) noexcept {
  poisoned_ = true;
  sequencer_.poison_epoch();
  return status.ok() ? Status::Internal("DeepSeek NCCL endpoint warm-up poisoned")
                     : status;
}

Status DeepSeekNcclEndpointWarmupRunner::start_phase(std::uint32_t phase) {
  const auto tokens = phase == 0 ? 1U : config_.maximum_token_count;
  const auto bytes = std::uint64_t{tokens} * 32768U;
  const auto& e = config_.endpoint;
  const auto role = e.communicator_local_rank == 0
                        ? DeepSeekNcclRole::kSend : DeepSeekNcclRole::kRecv;
  auto status = payload_->prepare(role, config_.device_buffer, bytes,
                                  config_.stream,
                                  (std::uint64_t{e.edge_id} << 32U) | phase + 1U);
  if (!status.ok()) return fail(status);
  status = transport_->bind_p2p(role, config_.device_buffer,
                                config_.buffer_capacity_bytes, config_.stream);
  if (!status.ok()) return fail(status);
  const auto ordinal = config_.first_operation_ordinal + phase;
  DeepSeekNcclP2pManifest manifest{
      ordinal, e.engine_epoch, e.communicator_generation, 1, ordinal, ordinal,
      e.edge_id, role, e.local_global_rank, e.peer_global_rank,
      e.communicator_local_rank, 1U - e.communicator_local_rank,
      config_.buffer_owner_id, 0, config_.buffer_capacity_bytes,
      config_.buffer_generation, e.context_identity, tokens};
  auto plan = DeepSeekNcclP2pPlan::Create(manifest);
  auto event = CompletionEventSlot::Create(config_.completion_events[phase],
                                            e.context_identity);
  auto frontier = CudaCompletionFrontier::Create(
      {e.engine_epoch, e.local_global_rank, e.communicator_generation,
       CudaCompletionPhase::kCopy, ordinal},
      ordinal, config_.submit_ns, config_.deadline_ns);
  if (!plan.ok()) return fail(plan.status());
  if (!event.ok()) return fail(event.status());
  if (!frontier.ok()) return fail(frontier.status());
  plan_ = std::move(*plan); event_ = std::move(*event);
  frontier_ = std::move(*frontier);
  status = sequencer_.issue(*plan_, *transport_);
  return status.ok() ? status : fail(status);
}

Status DeepSeekNcclEndpointWarmupRunner::begin() {
  if (begun_ || complete_ || poisoned_) {
    return Status::FailedPrecondition("DeepSeek NCCL endpoint warm-up cannot begin");
  }
  begun_ = true;
  return start_phase(0);
}

Result<std::optional<DeepSeekNcclEndpointWarmupReceipt>>
DeepSeekNcclEndpointWarmupRunner::poll(
    CompletionEventDriver& event_driver,
    CompletionEvidenceProvider& evidence, std::uint64_t now_ns) {
  if (!begun_ || complete_ || poisoned_ || !plan_ || !event_ || !frontier_) {
    return Status::FailedPrecondition("DeepSeek NCCL endpoint warm-up is not pollable");
  }
  if (now_ns < config_.submit_ns) {
    return fail(Status::InvalidArgument(
        "DeepSeek NCCL endpoint warm-up clock regressed"));
  }
  if (plan_->state() == DeepSeekNcclP2pState::kGroupEndPending) {
    auto status = plan_->poll_issue(*transport_);
    if (!status.ok()) return fail(status);
  }
  if (plan_->state() == DeepSeekNcclP2pState::kIssuedToStream) {
    auto status = plan_->record_completion(*event_, event_driver, config_.stream);
    if (!status.ok()) return fail(status);
  }
  if (plan_->state() == DeepSeekNcclP2pState::kDeviceInFlight) {
    auto status = plan_->poll_completion(*transport_, *event_, event_driver,
                                         *frontier_, evidence);
    if (!status.ok()) {
      if (status.code() == StatusCode::kUnavailable && !frontier_->poisoned() &&
          now_ns < config_.deadline_ns)
        return std::optional<DeepSeekNcclEndpointWarmupReceipt>{};
      if (status.code() == StatusCode::kUnavailable && !frontier_->poisoned())
        return fail(frontier_->expire(now_ns));
      return fail(status);
    }
  }
  if (plan_->state() != DeepSeekNcclP2pState::kCompleteVerified)
    return std::optional<DeepSeekNcclEndpointWarmupReceipt>{};
  auto digest = payload_->digest(config_.device_buffer, plan_->wire_bytes());
  if (!digest.ok()) return fail(digest.status());
  digests_[phase_] = *digest;
  auto status = sequencer_.close(*plan_);
  if (!status.ok()) return fail(status);
  status = event_->release(plan_->manifest().operation_ordinal);
  if (!status.ok()) return fail(status);
  if (phase_++ == 0) {
    status = start_phase(1);
    if (!status.ok()) return status;
    return std::optional<DeepSeekNcclEndpointWarmupReceipt>{};
  }
  complete_ = true;
  const auto& e = config_.endpoint;
  DeepSeekNcclEndpointBootstrapReceipt endpoint{
      e.engine_epoch, e.communicator_generation, e.bootstrap_lease_id,
      e.bootstrap_commitment_id, e.edge_id, e.local_global_rank,
      e.peer_global_rank, e.communicator_local_rank, e.device_identity,
      e.context_identity, e.config_identity, true, true};
  return std::optional<DeepSeekNcclEndpointWarmupReceipt>{{
      endpoint, 1, config_.maximum_token_count, digests_[0], digests_[1],
      true, true}};
}

}  // namespace pih
