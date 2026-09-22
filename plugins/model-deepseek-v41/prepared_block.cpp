#include "prepared_block.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
PreparedBlockOperation::PreparedBlockOperation(PreparedBlockOperation&& other) noexcept
    : config_(other.config_), launch_(other.launch_), weights_(std::move(other.weights_)), workspace_(other.workspace_),
      host_counts_(other.host_counts_), communicator_(other.communicator_), reservation_(other.reservation_),
      resources_(other.resources_), deadline_(other.deadline_), attention_(std::move(other.attention_)),
      ffn_(std::move(other.ffn_)), state_(other.state_) { other.workspace_ = nullptr; other.state_ = PreparedBlockState::kFailed; }
Result<PreparedBlockOperation> PreparedBlockOperation::Start(const FlashConfig& config, const PreparedBlockLaunch& x,
    std::span<const ExpertWeights> weights, ExpertWorkspaceOwner& workspace, EngramDeviceRegion host_counts,
    std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline,
    std::uint64_t reservation) {
  const auto attention = ValidateAttentionResidual(x.attention); if (!attention.ok()) return attention;
  const auto route = ValidateFfnRoute(config, x.ffn); if (!route.ok()) return route;
  const auto& a = x.attention.attention.assembly; const auto& o = x.attention.attention.output;
  const auto& d = x.ffn.dispatch; const auto& m = x.ffn.input.mix; const auto& router = x.ffn.router;
  if (d.world_size == 1 || weights.size() != 384U / d.world_size ||
      o.grouped.projection.groups != 8U / d.world_size || a.tokens != d.tokens || a.stream != d.stream ||
      a.ratio != config.attention_sharing()[d.layer].compression_ratio || !Same(a.error_flag, d.error_flag) ||
      !Same(x.attention.residual.output, m.residual) || !Same(x.attention_pre, x.ffn.input.input.collapse.pre))
    return Status::InvalidArgument("Prepared block layer, rank geometry or residual/pre connection mismatch");
  const auto binding = workspace.ValidateBinding(d.tokens, d.stream, resources.event, reservation); if (!binding.ok()) return binding;
  const auto& allocation = workspace.allocation_record(); const EngramDeviceRegion owned{allocation.address, allocation.bytes};
  if (x.tail.experts.merge.routed.address || x.tail.experts.merge.routed.bytes)
    return Status::InvalidArgument("Prepared block tail accumulator must be unbound");
  const auto accumulator = ExpertWorkspaceAccumulator(owned, d.tokens); if (!accumulator.ok()) return accumulator.status();
  auto tail = x.tail; tail.experts.merge.routed = *accumulator;
  const auto tail_valid = ValidateExpertResidual(tail); if (!tail_valid.ok()) return tail_valid;
  const auto& s = tail.experts.shared;
  std::vector<EngramDeviceRegion> future_reads{x.attention_pre, m.fn, m.scale, m.base, x.ffn.input.input.norm.weight,
      router.weight, router.bias, router.image_bias, router.image_mask,
      s.gate.weight, s.gate.weight_scales, s.up.weight, s.up.weight_scales, s.down.weight, s.down.weight_scales};
  for (const auto& expert : weights) for (const auto p : {expert.gate, expert.up, expert.down}) {
    for (const auto region : {p.weight, p.scales})
      if (!region.address || !region.bytes || region.bytes > std::numeric_limits<std::uintptr_t>::max() - region.address)
        return Status::InvalidArgument("Prepared block expert weight region invalid");
    future_reads.push_back(p.weight); future_reads.push_back(p.scales);
  }
  const std::array attention_writes{a.kv, a.indices, o.grouped.attention.output, o.grouped.inverse_rope.output,
      o.grouped.projection.output, o.linear.quantized, o.linear.activation_scales, o.linear.output, o.reduction,
      x.attention.residual.output, a.error_flag};
  for (const auto write : attention_writes) {
    if (Overlap(owned, write)) return Status::InvalidArgument("Attention writes into reserved expert workspace");
    for (const auto read : future_reads) if (Overlap(write, read))
      return Status::InvalidArgument("Attention overwrites FFN input coefficient or weight");
  }
  const std::array persistent_attention{a.window, a.compressed, o.grouped.attention.sink,
      o.grouped.projection.weight, o.linear.weight, o.linear.weight_scales};
  std::vector<EngramDeviceRegion> future_writes{owned, m.pre, m.post, m.comb, x.ffn.input.input.collapse.output,
      x.ffn.input.input.norm.output, router.logits, router.indices, router.route_weights, d.counts, d.slots,
      d.error_flag, tail.experts.merge.output, tail.residual.output};
  for (const auto* p : {&s.gate, &s.up, &s.down}) {
    future_writes.push_back(p->quantized); future_writes.push_back(p->activation_scales); future_writes.push_back(p->output);
  }
  future_writes.push_back(s.activation.output);
  for (const auto write : future_writes) for (const auto read : persistent_attention) if (Overlap(write, read))
    return Status::InvalidArgument("FFN overwrites retained attention cache or weight");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Prepared block deadline expired before submission");
  PreparedBlockOperation operation;
  operation.config_ = config; operation.launch_ = x; operation.weights_.assign(weights.begin(), weights.end());
  operation.workspace_ = &workspace; operation.host_counts_ = host_counts; operation.communicator_ = communicator;
  operation.resources_ = resources; operation.deadline_ = deadline;
  if (reservation) operation.reservation_ = reservation;
  else {
    const auto reserved = workspace.Reserve(); if (!reserved.ok()) return reserved.status();
    operation.reservation_ = *reserved;
  }
  auto submitted = AttentionTensorParallel::Start(x.attention, d.rank, communicator, resources, deadline);
  if (!submitted.ok()) return submitted.status();
  operation.attention_.emplace(std::move(*submitted)); operation.state_ = PreparedBlockState::kWaitingAttention;
  return operation;
}
Result<PreparedBlockState> PreparedBlockOperation::Advance() {
  if (state_ == PreparedBlockState::kFailed) return Status::FailedPrecondition("Prepared block failed or moved from");
  if (state_ == PreparedBlockState::kComplete) return state_;
  const auto previous = state_; state_ = PreparedBlockState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Prepared block deadline expired");
  if (previous == PreparedBlockState::kWaitingAttention) {
    const auto ready = attention_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != AttentionPipelineState::kComplete) { state_ = previous; return state_; }
    auto ffn = FfnOperation::Start(config_, launch_.ffn, weights_, *workspace_, launch_.tail,
        host_counts_, communicator_, resources_, deadline_, reservation_);
    if (!ffn.ok()) return ffn.status();
    ffn_.emplace(std::move(*ffn)); state_ = PreparedBlockState::kWaitingFfn; return state_;
  }
  const auto ready = ffn_->Advance(); if (!ready.ok()) return ready.status();
  state_ = *ready == FfnOperationState::kComplete ? PreparedBlockState::kComplete : PreparedBlockState::kWaitingFfn;
  return state_;
}
}  // namespace pih::deepseek_v41
