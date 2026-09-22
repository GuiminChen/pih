#include "pih/model/deepseek_nccl_generation_warmup_runner.h"

#include <limits>

namespace pih {

Result<DeepSeekNcclGenerationWarmupRunner>
DeepSeekNcclGenerationWarmupRunner::Create(
    std::vector<DeepSeekNcclGenerationWarmupEndpointResources> endpoints) {
  if (endpoints.empty() || endpoints.size() > 6 ||
      endpoints.size() % 2 != 0) {
    return Status::InvalidArgument(
        "DeepSeek NCCL generation warm-up endpoint count is invalid");
  }
  for (std::size_t index = 0; index < endpoints.size(); ++index) {
    const auto& endpoint = endpoints[index];
    if (endpoint.first_operation_ordinal == 0 ||
        endpoint.first_operation_ordinal ==
            std::numeric_limits<std::uint64_t>::max() ||
        endpoint.buffer_owner_id == 0 || endpoint.buffer_generation == 0 ||
        endpoint.device_buffer == nullptr || endpoint.buffer_capacity_bytes == 0 ||
        endpoint.stream == 0 || endpoint.completion_events[0] == 0 ||
        endpoint.completion_events[1] == 0 ||
        endpoint.submit_ns >= endpoint.deadline_ns ||
        endpoint.payload == nullptr || endpoint.event_driver == nullptr ||
        endpoint.evidence == nullptr || endpoint.clock == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek NCCL generation warm-up endpoint resource is invalid");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      const auto prior_first = endpoints[prior].first_operation_ordinal;
      if (endpoint.first_operation_ordinal == prior_first ||
          endpoint.first_operation_ordinal == prior_first + 1 ||
          endpoint.first_operation_ordinal + 1 == prior_first) {
        return Status::FailedPrecondition(
            "DeepSeek NCCL generation warm-up reuses an operation ordinal");
      }
    }
  }
  return DeepSeekNcclGenerationWarmupRunner(std::move(endpoints));
}

Result<std::vector<DeepSeekNcclWarmupReceipt>>
DeepSeekNcclGenerationWarmupRunner::run(
    DeepSeekNcclCommunicatorGeneration& generation,
    const DeepSeekNcclEndpointManifestPlan& manifests,
    std::uint32_t maximum_wire_tokens) {
  const auto& identity = generation.identity();
  if (consumed_ ||
      generation.state() != DeepSeekNcclGenerationState::kReconciled ||
      maximum_wire_tokens == 0 || manifests.world_size() != identity.world_size ||
      manifests.edges().size() + 1 != identity.world_size ||
      endpoints_.size() != manifests.edges().size() * 2) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation warm-up topology is not runnable");
  }
  consumed_ = true;
  auto gate = DeepSeekNcclBoundaryWarmupGate::Create(
      manifests, maximum_wire_tokens);
  if (!gate.ok()) return gate.status();
  std::vector<DeepSeekNcclEndpointWarmupRunner> runners;
  runners.reserve(endpoints_.size());
  for (std::uint32_t edge = 0; edge < manifests.edges().size(); ++edge) {
    for (std::uint32_t local = 0; local < 2; ++local) {
      const auto index = static_cast<std::size_t>(edge) * 2 + local;
      const auto& resources = endpoints_[index];
      const auto& endpoint = local == 0 ? manifests.edges()[edge].lower
                                       : manifests.edges()[edge].upper;
      auto* transport = local == 0
          ? generation.outgoing_warmup_transport(edge)
          : generation.incoming_warmup_transport(edge + 1);
      if (transport == nullptr) {
        (void)generation.abort();
        return Status::FailedPrecondition(
            "DeepSeek NCCL reconciled warm-up transport is unavailable");
      }
      auto runner = DeepSeekNcclEndpointWarmupRunner::Create(
          {endpoint, maximum_wire_tokens,
           resources.first_operation_ordinal, resources.buffer_owner_id,
           resources.buffer_generation, resources.device_buffer,
           resources.buffer_capacity_bytes, resources.stream,
           resources.completion_events, resources.submit_ns,
           resources.deadline_ns},
          *transport, *resources.payload);
      if (!runner.ok()) {
        (void)generation.abort();
        return runner.status();
      }
      runners.push_back(std::move(*runner));
    }
  }
  for (auto& runner : runners) {
    auto begun = runner.begin();
    if (!begun.ok()) {
      (void)generation.abort();
      return begun;
    }
  }
  std::vector<bool> complete(runners.size(), false);
  std::size_t remaining = runners.size();
  while (remaining != 0) {
    for (std::size_t index = 0; index < runners.size(); ++index) {
      if (complete[index]) continue;
      auto now = endpoints_[index].clock->now_ns();
      if (!now.ok()) {
        (void)generation.abort();
        return now.status();
      }
      auto receipt = runners[index].poll(
          *endpoints_[index].event_driver, *endpoints_[index].evidence, *now);
      if (!receipt.ok()) {
        (void)generation.abort();
        return receipt.status();
      }
      if (!receipt->has_value()) continue;
      auto accepted = gate->accept(std::move(**receipt));
      if (!accepted.ok()) {
        (void)generation.abort();
        return accepted;
      }
      complete[index] = true;
      --remaining;
    }
  }
  auto receipts = gate->finalize();
  if (!receipts.ok()) {
    (void)generation.abort();
    return receipts.status();
  }
  return receipts;
}

}  // namespace pih
