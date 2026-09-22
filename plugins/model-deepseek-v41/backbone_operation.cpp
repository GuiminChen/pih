#include "backbone_operation.h"
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
namespace {
constexpr std::array<unsigned, 8> kIndexLayers{2, 8, 14, 20, 24, 28, 32, 36};
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
BackboneOperation::BackboneOperation(const FlashConfig& config, BlockSequence& sequence,
    const BackboneResources& resources, Clock::time_point deadline)
    : config_(config), sequence_(sequence), resources_(resources), deadline_(deadline),
      start_(sequence.start_), tokens_(sequence.input_tokens_) {}
BackboneOperation::~BackboneOperation() {
  if (state_ == BackboneState::kRunning) sequence_.Fail();
}
Result<std::unique_ptr<BackboneOperation>> BackboneOperation::Start(const FlashConfig& config,
    BlockSequence& sequence, const BackboneResources& r, Clock::time_point deadline) {
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Backbone deadline expired before admission");
  if (sequence.failed_ || sequence.active_ || sequence.layer_ || !sequence.input_ready_ ||
      config.config_sha256() != sequence.config_.config_sha256())
    return Status::FailedPrecondition("Backbone requires completed embedding at layer zero");
  const std::array regions{r.phase_arena, r.engram_arena, r.cache_arena, r.ffn_arenas[0], r.ffn_arenas[1],
      r.attention_arena, r.output_arena, r.compressor_arena,
      r.indexer_arenas[0], r.indexer_arenas[1], r.indexer_arenas[2], r.indexer_arenas[3],
      r.indexer_arenas[4], r.indexer_arenas[5], r.indexer_arenas[6], r.indexer_arenas[7]};
  const std::array sizes{r.phases.bytes(), r.engram.bytes(), r.cache.bytes(), r.ffn.bytes(), r.ffn.bytes(),
      r.attention.bytes(), r.output.bytes(), r.compressor.bytes(), r.indexer.bytes(), r.indexer.bytes(),
      r.indexer.bytes(), r.indexer.bytes(), r.indexer.bytes(), r.indexer.bytes(), r.indexer.bytes(), r.indexer.bytes()};
  const auto& allocation = r.workspace.allocation_record();
  const EngramDeviceRegion expert{allocation.address, allocation.bytes};
  if (!Valid(expert)) return Status::InvalidArgument("Backbone expert workspace is absent or invalid");
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (!Valid(regions[i]) || regions[i].address % 256 || regions[i].bytes != sizes[i] || Overlap(regions[i], expert))
      return Status::InvalidArgument("Backbone arena extent, alignment or expert overlap invalid");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(regions[i], regions[j]))
      return Status::InvalidArgument("Backbone arenas must be disjoint");
  }
  const auto weights = r.weights.ValidateScratch(regions); if (!weights.ok()) return weights;
  try {
    auto operation = std::unique_ptr<BackboneOperation>(new BackboneOperation(config, sequence, r, deadline));
    const auto started = operation->StartLayer(); if (!started.ok()) return started;
    operation->state_ = BackboneState::kRunning;
    return operation;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Backbone operation allocation failed");
  }
}
Status BackboneOperation::StartLayer() {
  const auto& r = resources_;
  EngramDeviceRegion indexer;
  for (std::size_t i = 0; i < kIndexLayers.size(); ++i)
    if (layer_ == kIndexLayers[i]) indexer = r.indexer_arenas[i];
  auto block = BlockOperation::Start(config_, sequence_, r.phases, r.phase_arena, r.engram, r.engram_arena,
      r.weights, r.workspace, r.cache, r.cache_arena, r.ffn, r.ffn_arenas[layer_ % 2],
      r.attention, r.attention_arena, r.output, r.output_arena, r.compressor, r.compressor_arena,
      r.indexer, indexer, r.host_counts, r.communicator, r.completion, deadline_);
  if (!block.ok()) return block.status();
  block_.emplace(std::move(*block)); return Status::Ok();
}
Result<BackboneState> BackboneOperation::Advance() {
  try { return AdvanceImpl(); }
  catch (const std::bad_alloc&) {
    state_ = BackboneState::kFailed; sequence_.Fail();
    return Status::ResourceExhausted("Backbone continuation allocation failed");
  }
}
Result<BackboneState> BackboneOperation::AdvanceImpl() {
  if (state_ == BackboneState::kFailed) return Status::FailedPrecondition("Backbone operation failed");
  if (state_ == BackboneState::kComplete) return state_;
  // Any error after starting the step retires the entire sequence, even when
  // it occurs between layers before the next layer enqueues work.
  const auto fail = [&](Status status) -> Result<BackboneState> {
    state_ = BackboneState::kFailed; sequence_.Fail(); return status;
  };
  if (Clock::now() >= deadline_) return fail(Status::DeadlineExceeded("Backbone deadline expired"));
  if (sequence_.failed_ || sequence_.layer_ != layer_ || sequence_.start_ != start_)
    return fail(Status::FailedPrecondition("Backbone sequence was changed outside the active operation"));
  auto ready = block_->Advance(); if (!ready.ok()) return fail(ready.status());
  if (*ready != BlockOperationState::kComplete) return state_;
  block_.reset();
  if (++layer_ == FlashConfig::kMainLayers) {
    auto output = sequence_.StepOutput(); if (!output.ok()) return fail(output.status());
    if (output->start != start_ || output->tokens != tokens_)
      return fail(Status::Internal("Backbone output step mismatch"));
    state_ = BackboneState::kComplete; return state_;
  }
  if (resources_.workspace.state() != ExpertWorkspaceState::kReady)
    return fail(Status::FailedPrecondition("Completed block did not retire expert workspace"));
  const auto started = StartLayer(); if (!started.ok()) return fail(started);
  return state_;
}
Result<SequenceStepOutput> BackboneOperation::Output() const {
  if (state_ != BackboneState::kComplete) return Status::FailedPrecondition("Backbone output is not complete");
  auto output = sequence_.StepOutput(); if (!output.ok()) return output.status();
  if (output->start != start_ || output->tokens != tokens_)
    return Status::FailedPrecondition("Backbone output was superseded by another step");
  return output;
}
}  // namespace pih::deepseek_v41
