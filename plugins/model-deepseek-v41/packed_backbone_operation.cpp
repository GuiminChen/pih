#include "packed_backbone_operation.h"

namespace pih::deepseek_v41 {
namespace {
constexpr std::array<unsigned, 8> kIndexLayers{2, 8, 14, 20, 24, 28, 32, 36};
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
std::array<EngramDeviceRegion, 16> PrivateRegions(const BackboneResources& r) {
  std::array<EngramDeviceRegion, 16> result{r.phase_arena, r.engram_arena, r.cache_arena,
      r.ffn_arenas[0], r.ffn_arenas[1], r.attention_arena, r.output_arena, r.compressor_arena};
  std::copy(r.indexer_arenas.begin(), r.indexer_arenas.end(), result.begin() + 8);
  return result;
}
}
PackedBackboneOperation::~PackedBackboneOperation() { if (state_ == BackboneState::kRunning) Fail(); }
void PackedBackboneOperation::Fail() noexcept {
  state_ = BackboneState::kFailed;
  for (auto* member : sequences_) member->Fail();
}
Result<std::unique_ptr<PackedBackboneOperation>> PackedBackboneOperation::Start(const FlashConfig& config,
    const PackedRequest& plan, std::span<BlockSequence* const> slots,
    const PackedBackboneResources& resources, Clock::time_point deadline) {
  if (plan.spans().empty() || resources.members.size() != plan.spans().size() || Clock::now() >= deadline)
    return Status::InvalidArgument("Packed backbone resource count or deadline invalid");
  const auto total = plan.tokens().size();
  if (!resources.attention_residual.address || resources.attention_residual.bytes != total * 4ULL * 5120 * 2 ||
      !resources.attention_pre.address || resources.attention_pre.bytes != total * 4ULL * 4 ||
      resources.attention_residual.bytes > UINTPTR_MAX - resources.attention_residual.address ||
      resources.attention_pre.bytes > UINTPTR_MAX - resources.attention_pre.address ||
      Overlap(resources.attention_residual, resources.attention_pre))
    return Status::InvalidArgument("Packed backbone staging row extents invalid");
  const auto& first = resources.members.front();
  const std::array<EngramDeviceRegion, 4> shared{resources.ffn_arenas[0], resources.ffn_arenas[1],
      resources.attention_residual, resources.attention_pre};
  for (unsigned i = 0; i < shared.size(); ++i) {
    if (!shared[i].address || !shared[i].bytes || shared[i].bytes > UINTPTR_MAX - shared[i].address)
      return Status::InvalidArgument("Packed backbone shared allocation extent invalid");
    for (unsigned j = 0; j < i; ++j) if (Overlap(shared[i], shared[j]))
      return Status::InvalidArgument("Packed backbone shared arenas alias");
  }
  for (unsigned i = 0; i < resources.members.size(); ++i) {
    const auto& member = resources.members[i]; const auto& span = plan.spans()[i];
    const auto private_regions = PrivateRegions(member);
    for (const auto region : private_regions) {
      if (!region.address || !region.bytes || region.bytes > UINTPTR_MAX - region.address)
        return Status::InvalidArgument("Packed backbone member allocation extent invalid");
      for (const auto common : shared) if (Overlap(region, common))
        return Status::InvalidArgument("Packed backbone private and shared arenas alias");
      for (unsigned prior = 0; prior < i; ++prior)
        for (const auto other : PrivateRegions(resources.members[prior])) if (Overlap(region, other))
          return Status::InvalidArgument("Packed backbone private member arenas alias");
    }
    if (span.finish || span.cache_slot >= slots.size() || !slots[span.cache_slot])
      return Status::InvalidArgument("Packed backbone member slot absent");
    const auto& sequence = *slots[span.cache_slot];
    if (sequence.failed_ || sequence.active_ || sequence.layer_ || !sequence.input_ready_ ||
        sequence.start_ != span.position || sequence.input_tokens_ != span.token_count ||
        config.config_sha256() != sequence.config_.config_sha256() ||
        &member.workspace != &first.workspace || &member.weights != &first.weights ||
        member.communicator != first.communicator || member.completion.event != first.completion.event ||
        member.host_counts.address != first.host_counts.address)
      return Status::FailedPrecondition("Packed backbone member state or shared execution owner differs");
    for (unsigned previous = 0; previous < i; ++previous) {
      if (slots[plan.spans()[previous].cache_slot] == &sequence ||
          Overlap(member.cache_arena, resources.members[previous].cache_arena))
        return Status::InvalidArgument("Packed backbone sequence caches alias");
    }
  }
  auto operation = std::unique_ptr<PackedBackboneOperation>(new PackedBackboneOperation(config, plan, resources, deadline));
  for (const auto& span : plan.spans()) operation->sequences_.push_back(slots[span.cache_slot]);
  operation->sources_.reserve(plan.spans().size());
  operation->state_ = BackboneState::kRunning;
  const auto started = operation->StartLayer(); if (!started.ok()) { operation->Fail(); return started; }
  return operation;
}
Status PackedBackboneOperation::StartLayer() {
  sources_.clear(); block_.reset(); source_member_ = 0;
  auto reserved = resources_.members.front().workspace.Reserve(); if (!reserved.ok()) return reserved.status();
  reservation_ = *reserved; return StartSources();
}
Status PackedBackboneOperation::StartSources() {
  const auto& r = resources_.members[source_member_];
  EngramDeviceRegion indexer;
  for (unsigned i = 0; i < kIndexLayers.size(); ++i) if (layer_ == kIndexLayers[i]) indexer = r.indexer_arenas[i];
  auto source = BlockOperation::Start(config_, *sequences_[source_member_], r.phases, r.phase_arena, r.engram, r.engram_arena,
      r.weights, r.workspace, r.cache, r.cache_arena, r.ffn, r.ffn_arenas[layer_ % 2],
      r.attention, r.attention_arena, r.output, r.output_arena, r.compressor, r.compressor_arena,
      r.indexer, indexer, r.host_counts, r.communicator, r.completion, deadline_, true, reservation_);
  if (!source.ok()) return source.status();
  sources_.push_back(std::move(*source)); return Status::Ok();
}
Result<BackboneState> PackedBackboneOperation::Advance() {
  try { auto result = AdvanceImpl(); if (!result.ok()) Fail(); return result; }
  catch (const std::bad_alloc&) { Fail(); return Status::ResourceExhausted("Packed backbone continuation allocation failed"); }
}
Result<BackboneState> PackedBackboneOperation::AdvanceImpl() {
  if (state_ != BackboneState::kRunning) return state_ == BackboneState::kComplete ? Result<BackboneState>(state_) :
      Result<BackboneState>(Status::FailedPrecondition("Packed backbone failed"));
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Packed backbone deadline elapsed");
  if (!block_) {
    auto ready = sources_.back().Advance(); if (!ready.ok()) return ready.status();
    if (*ready != BlockOperationState::kSourcesReady) return state_;
    if (++source_member_ < sequences_.size()) {
      const auto next = StartSources(); if (!next.ok()) return next; return state_;
    }
    const auto& r = resources_.members.front(); const auto& allocation = r.workspace.allocation_record();
    auto accumulator = ExpertWorkspaceAccumulator({allocation.address, allocation.bytes}, plan_.tokens().size());
    if (!accumulator.ok()) return accumulator.status();
    auto views = resources_.ffn.Bind(r.weights, config_, resources_.ffn_arenas[layer_ % 2], layer_, plan_.tokens().size(),
        resources_.attention_residual, resources_.attention_pre, *accumulator, sequences_.front()->error_, sequences_.front()->stream_);
    if (!views.ok()) return views.status(); ffn_views_ = *views;
    auto weights = UploadedRoutedExperts(r.weights, config_, views->route.dispatch); if (!weights.ok()) return weights.status();
    std::vector<AttentionResidualLaunch> attention; std::vector<EngramDeviceRegion> pre;
    for (const auto& source : sources_) {
      auto prepared = source.PreparedLaunch(); if (!prepared.ok()) return prepared.status();
      attention.push_back(prepared->attention); pre.push_back(prepared->attention_pre);
    }
    auto submitted = PreparedBlockOperation::StartPacked(config_, plan_, attention, pre, resources_.attention_pre,
        views->route, views->tail, *weights, r.workspace, r.host_counts, r.communicator, r.completion, deadline_, reservation_);
    if (!submitted.ok()) return submitted.status(); block_.emplace(std::move(*submitted)); return state_;
  }
  auto ready = block_->Advance(); if (!ready.ok()) return ready.status();
  if (*ready != PreparedBlockState::kComplete) return state_;
  for (unsigned i = 0; i < sequences_.size(); ++i) {
    const auto& span = plan_.spans()[i];
    const EngramDeviceRegion residual{ffn_views_.tail.residual.output.address + span.token_offset * 4ULL * 5120 * 2,
        span.token_count * 4ULL * 5120 * 2};
    const EngramDeviceRegion pre{ffn_views_.route.input.mix.pre.address + span.token_offset * 4ULL * 4,
        span.token_count * 4ULL * 4};
    const auto published = sources_[i].CompletePacked(*block_, residual, pre); if (!published.ok()) return published;
  }
  if (++layer_ == FlashConfig::kMainLayers) { state_ = BackboneState::kComplete; return state_; }
  const auto next = StartLayer(); if (!next.ok()) return next; return state_;
}
Result<SequenceStepOutput> PackedBackboneOperation::Output(std::uint32_t member) const {
  if (state_ != BackboneState::kComplete || member >= sequences_.size())
    return Status::FailedPrecondition("Packed backbone member output not complete");
  return sequences_[member]->StepOutput();
}
}
