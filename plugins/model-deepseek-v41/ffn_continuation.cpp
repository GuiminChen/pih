#include "ffn_continuation.h"
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
template<class Expert>
void Buffers(const Expert& s, std::vector<EngramDeviceRegion>& weights, std::vector<EngramDeviceRegion>& writes) {
  for (const auto* p : {&s.gate, &s.up, &s.down}) {
    weights.push_back(p->weight); weights.push_back(p->weight_scales);
    writes.push_back(p->quantized); writes.push_back(p->activation_scales); writes.push_back(p->output);
  }
  writes.push_back(s.activation.output);
}
}
Status ValidateFfnContinuation(const FlashConfig& config, const FfnRouteLaunch& route,
    const ExpertCounts& counts, std::span<const ExpertTokenChainLaunch> batch, const ExpertResidualLaunch& tail) {
  const auto routed = ValidateFfnRoute(config, route); if (!routed.ok()) return routed;
  const auto observed = counts.ValidatePlan(route.dispatch); if (!observed.ok()) return observed;
  const auto batched = ValidateExpertBatch(batch, counts); if (!batched.ok()) return batched;
  const auto ended = ValidateExpertResidual(tail); if (!ended.ok()) return ended;
  const auto& m = route.input.mix; const auto& n = route.input.input.norm;
  const auto& s = tail.experts.shared; const auto& merge = tail.experts.merge; const auto& r = tail.residual;
  if (!Same(batch.front().gather.input, n.output) || !Same(batch.front().gather.route_weights, route.router.route_weights) ||
      !Same(batch.front().accumulator, merge.routed) || !Same(s.gate.input, n.output) ||
      !Same(r.residual, m.residual) || !Same(r.post, m.post) || !Same(r.comb, m.comb) ||
      !Same(r.error_flag, m.error_flag) || r.stream != m.stream || r.tokens != m.tokens)
    return Status::InvalidArgument("FFN continuation input, accumulator or coefficient connection mismatch");
  std::vector<EngramDeviceRegion> weights, batch_writes{merge.routed, m.error_flag}, tail_writes{merge.output, r.output, m.error_flag};
  Buffers(s, weights, tail_writes);
  for (const auto& expert : batch) {
    batch_writes.push_back(expert.gather.output); batch_writes.push_back(expert.gather.gathered_weights);
    if (expert.expert) Buffers(*expert.expert, weights, batch_writes);
  }
  const std::array retained{m.residual, m.pre, m.post, m.comb, n.output};
  for (const auto write : batch_writes) {
    for (const auto read : retained) if (Overlap(write, read))
      return Status::InvalidArgument("FFN batch overwrites retained residual, coefficients or hidden input");
    for (const auto weight : weights) if (Overlap(write, weight))
      return Status::InvalidArgument("FFN batch overwrites shared or routed expert weight");
  }
  // pre is returned to the next sublayer, so even final residual writes must preserve it.
  for (const auto write : tail_writes) {
    if (Overlap(write, m.pre)) return Status::InvalidArgument("FFN tail overwrites next-sublayer pre coefficients");
    for (const auto weight : weights) if (Overlap(write, weight))
      return Status::InvalidArgument("FFN tail overwrites persistent expert weight");
  }
  const std::array route_writes{m.pre, m.post, m.comb, route.input.input.collapse.output, n.output,
      route.router.logits, route.router.indices, route.router.route_weights, route.dispatch.counts,
      route.dispatch.slots, m.error_flag};
  for (const auto write : route_writes) for (const auto weight : weights) if (Overlap(write, weight))
    return Status::InvalidArgument("FFN route buffers overlap later expert weights");
  return Status::Ok();
}
FfnContinuation::FfnContinuation(FfnContinuation&& other) noexcept
    : tail_launch_(other.tail_launch_), world_(other.world_), rank_(other.rank_), communicator_(other.communicator_),
      resources_(other.resources_), deadline_(other.deadline_), batch_(std::move(other.batch_)),
      tail_(std::move(other.tail_)), workspace_(other.workspace_), reservation_(other.reservation_), state_(other.state_) {
  other.workspace_ = nullptr; other.state_ = FfnContinuationState::kFailed;
}
Result<FfnContinuation> FfnContinuation::Start(const FlashConfig& config, const FfnRouteLaunch& route,
    const ExpertCounts& counts, std::span<const ExpertWeights> weights, ExpertWorkspaceOwner& workspace,
    const ExpertResidualLaunch& tail_template, std::uintptr_t communicator,
    const EngramCompletionResources& resources, Clock::time_point deadline, std::uint64_t reservation) {
  const auto& d = route.dispatch;
  if (d.world_size == 1) return Status::InvalidArgument("FFN multi-rank continuation requires 2, 4 or 8 ranks");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("FFN continuation deadline expired");
  const auto binding = workspace.ValidateBinding(d.tokens, d.stream, resources.event, reservation); if (!binding.ok()) return binding;
  if (tail_template.experts.merge.routed.address || tail_template.experts.merge.routed.bytes)
    return Status::InvalidArgument("FFN tail routed input must be absent; workspace binding supplies it");
  const auto& allocation = workspace.allocation_record();
  auto built = BuildExpertWorkspaceBatch(d, counts, route.input.input.norm.output, route.router.route_weights,
      weights, {allocation.address, allocation.bytes});
  if (!built.ok()) return built.status();
  auto tail = tail_template; tail.experts.merge.routed = built->accumulator;
  const auto validation = ValidateFfnContinuation(config, route, counts, built->experts, tail); if (!validation.ok()) return validation;
  // The entire owned allocation (including unused row capacity) is reusable
  // after completion. No externally retained tail/route buffer may live in it.
  const EngramDeviceRegion owned{allocation.address, allocation.bytes};
  const auto& s = tail.experts.shared; const auto& r = tail.residual; const auto& m = route.input.mix;
  const std::array retained{m.residual, m.fn, m.scale, m.base, m.pre, m.post, m.comb,
      route.input.input.collapse.pre, route.input.input.collapse.output, route.input.input.norm.weight,
      route.router.weight, route.router.bias, route.router.image_bias, route.router.image_mask, route.router.logits,
      s.gate.input, s.gate.weight, s.gate.weight_scales, s.up.weight, s.up.weight_scales,
      s.down.weight, s.down.weight_scales, s.gate.quantized, s.gate.activation_scales, s.gate.output,
      s.up.quantized, s.up.activation_scales, s.up.output, s.activation.output,
      s.down.quantized, s.down.activation_scales, s.down.output, tail.experts.merge.output,
      r.residual, r.post, r.comb, r.output, r.error_flag};
  for (const auto region : retained) if (Overlap(owned, region))
    return Status::InvalidArgument("Owned expert workspace overlaps externally retained FFN storage");
  const auto transport = ValidateFp32Reduction({tail.experts.merge.routed, d.stream, d.world_size, d.rank}, communicator);
  if (!transport.ok()) return transport;
  const auto resources_ready = ValidateEngramCompletionResources(resources); if (!resources_ready.ok()) return resources_ready;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("FFN continuation deadline expired before workspace use");
  const auto active = workspace.BeginUse(reservation); if (!active.ok()) return active.status();
  auto operation_batch = ExpertBatch::Start(built->experts, counts, resources, deadline); if (!operation_batch.ok()) return operation_batch.status();
  FfnContinuation operation;
  operation.tail_launch_ = tail; operation.world_ = d.world_size; operation.rank_ = d.rank;
  operation.communicator_ = communicator; operation.resources_ = resources; operation.deadline_ = deadline;
  operation.workspace_ = &workspace;
  operation.reservation_ = reservation;
  operation.batch_.emplace(std::move(*operation_batch)); operation.state_ = FfnContinuationState::kWaitingBatch;
  return operation;
}
Result<FfnContinuationState> FfnContinuation::Advance() {
  if (state_ == FfnContinuationState::kFailed) return Status::FailedPrecondition("FFN continuation failed or moved from");
  if (state_ == FfnContinuationState::kComplete) return state_;
  const auto previous = state_; state_ = FfnContinuationState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("FFN continuation deadline expired");
  if (previous == FfnContinuationState::kWaitingRetirement) {
    const auto retired = workspace_->PollRetirement(); if (!retired.ok()) return retired.status();
    if (Clock::now() >= deadline_) return Status::DeadlineExceeded("FFN deadline expired at workspace retirement");
    state_ = *retired ? FfnContinuationState::kComplete : FfnContinuationState::kWaitingRetirement;
    return state_;
  }
  if (previous == FfnContinuationState::kWaitingBatch) {
    const auto ready = batch_->Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) { state_ = previous; return state_; }
    auto tail = ExpertTensorParallel::Start(tail_launch_, world_, rank_, communicator_, resources_, deadline_);
    if (!tail.ok()) return tail.status();
    tail_.emplace(std::move(*tail)); state_ = FfnContinuationState::kWaitingTail; return state_;
  }
  const auto ready = tail_->Advance(); if (!ready.ok()) return ready.status();
  if (*ready == ExpertPipelineState::kComplete) {
    const auto fenced = workspace_->RecordRetirementFence(reservation_); if (!fenced.ok()) return fenced;
    state_ = FfnContinuationState::kWaitingRetirement;
  } else state_ = FfnContinuationState::kWaitingTail;
  return state_;
}
}  // namespace pih::deepseek_v41
