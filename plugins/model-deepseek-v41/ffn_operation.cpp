#include "ffn_operation.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Valid(EngramDeviceRegion r, std::uint64_t bytes) {
  return r.address && r.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
}
FfnOperation::FfnOperation(FfnOperation&& other) noexcept
    : config_(other.config_), route_(other.route_), tail_(other.tail_), weights_(std::move(other.weights_)),
      workspace_(other.workspace_), reservation_(other.reservation_), communicator_(other.communicator_), resources_(other.resources_),
      deadline_(other.deadline_), counts_(std::move(other.counts_)), execution_(std::move(other.execution_)), state_(other.state_) {
  other.workspace_ = nullptr; other.state_ = FfnOperationState::kFailed;
}
Result<FfnOperation> FfnOperation::Start(const FlashConfig& config, const FfnRouteLaunch& route,
    std::span<const ExpertWeights> weights, ExpertWorkspaceOwner& workspace,
    const ExpertResidualLaunch& tail_template, EngramDeviceRegion host_counts,
    std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline,
    std::uint64_t reservation) {
  const auto validated = ValidateFfnRoute(config, route); if (!validated.ok()) return validated;
  const auto& d = route.dispatch;
  if (d.world_size == 1 || weights.size() != 384U / d.world_size)
    return Status::InvalidArgument("FFN operation requires all local weights on 2, 4 or 8 ranks");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("FFN operation deadline expired");
  const auto binding = workspace.ValidateBinding(d.tokens, d.stream, resources.event, reservation); if (!binding.ok()) return binding;
  if (tail_template.experts.merge.routed.address || tail_template.experts.merge.routed.bytes)
    return Status::InvalidArgument("FFN operation requires an unbound routed tail input");
  const auto& allocation = workspace.allocation_record();
  const EngramDeviceRegion owned{allocation.address, allocation.bytes};
  const auto accumulator = ExpertWorkspaceAccumulator(owned, d.tokens); if (!accumulator.ok()) return accumulator.status();
  auto bound_tail = tail_template; bound_tail.experts.merge.routed = *accumulator;
  const auto tail_valid = ValidateExpertResidual(bound_tail); if (!tail_valid.ok()) return tail_valid;
  const auto& m = route.input.mix; const auto& s = bound_tail.experts.shared;
  std::vector<EngramDeviceRegion> weight_regions{s.gate.weight, s.gate.weight_scales,
      s.up.weight, s.up.weight_scales, s.down.weight, s.down.weight_scales};
  for (const auto& expert : weights) for (const auto projection : {expert.gate, expert.up, expert.down}) {
    if (!Valid(projection.weight, 5120ULL * 2304 / 2) || !Valid(projection.scales, 5120ULL * 2304 / 32))
      return Status::InvalidArgument("FFN operation expert weight layout invalid");
    weight_regions.push_back(projection.weight); weight_regions.push_back(projection.scales);
  }
  const std::array route_writes{m.pre, m.post, m.comb, route.input.input.collapse.output,
      route.input.input.norm.output, route.router.logits, route.router.indices, route.router.route_weights,
      d.counts, d.slots, d.error_flag};
  for (const auto weight : weight_regions) {
    if (Overlap(owned, weight)) return Status::InvalidArgument("FFN workspace overlaps persistent expert weight");
    for (const auto write : route_writes) if (Overlap(write, weight))
      return Status::InvalidArgument("FFN route overwrites future expert weights");
  }
  for (const auto write : route_writes) if (Overlap(owned, write))
    return Status::InvalidArgument("FFN route writes into reserved expert workspace");
  const auto transport = ValidateFp32Reduction({*accumulator, d.stream, d.world_size, d.rank}, communicator);
  if (!transport.ok()) return transport;
  FfnOperation operation;
  operation.config_ = config; operation.route_ = route; operation.tail_ = tail_template;
  operation.weights_.assign(weights.begin(), weights.end()); operation.workspace_ = &workspace;
  operation.communicator_ = communicator; operation.resources_ = resources; operation.deadline_ = deadline;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("FFN deadline expired before workspace reservation");
  if (reservation) operation.reservation_ = reservation;
  else {
    const auto reserved = workspace.Reserve(); if (!reserved.ok()) return reserved.status();
    operation.reservation_ = *reserved;
  }
  auto counts = LaunchFfnRoute(config, route, host_counts, resources, deadline); if (!counts.ok()) return counts.status();
  operation.counts_.emplace(std::move(*counts)); operation.state_ = FfnOperationState::kWaitingCounts;
  return operation;
}
Result<FfnOperationState> FfnOperation::Advance() {
  if (state_ == FfnOperationState::kFailed) return Status::FailedPrecondition("FFN operation failed or moved from");
  if (state_ == FfnOperationState::kComplete) return state_;
  const auto previous = state_; state_ = FfnOperationState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("FFN operation deadline expired");
  if (previous == FfnOperationState::kWaitingCounts) {
    const auto ready = counts_->Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) { state_ = previous; return state_; }
    auto execution = FfnContinuation::Start(config_, route_, *counts_, weights_, *workspace_, tail_, communicator_, resources_, deadline_, reservation_);
    if (!execution.ok()) return execution.status();
    execution_.emplace(std::move(*execution)); state_ = FfnOperationState::kWaitingExecution; return state_;
  }
  const auto ready = execution_->Advance(); if (!ready.ok()) return ready.status();
  state_ = *ready == FfnContinuationState::kComplete ? FfnOperationState::kComplete : FfnOperationState::kWaitingExecution;
  return state_;
}
}  // namespace pih::deepseek_v41
