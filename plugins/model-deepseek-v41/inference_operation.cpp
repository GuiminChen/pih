#include "inference_operation.h"
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
InferenceOperation::InferenceOperation(const FlashConfig& config, BlockSequence& sequence,
    const BackboneResources& resources, const BoundaryViews& views,
    const SamplingIdentity& identity, Clock::time_point deadline)
    : config_(config), sequence_(sequence), resources_(resources), views_(views), identity_(identity), deadline_(deadline) {
  resources_.host_counts = views.host_counts;
  resources_.completion.host_error_flag = views.host_error;
  step_end_ = sequence.start_ + views.embedding.tokens;
}
InferenceOperation::~InferenceOperation() {
  if (state_ != InferenceState::kFailed && state_ != InferenceState::kComplete) sequence_.Fail();
}
Result<std::unique_ptr<InferenceOperation>> InferenceOperation::Start(const FlashConfig& config,
    BlockSequence& sequence, EngramHashState& hashes, std::span<const std::uint32_t> tokens,
    const BoundaryPlan& boundary, EngramDeviceRegion device, EngramDeviceRegion host,
    std::uintptr_t stream, const BackboneResources& resources,
    const SamplingParameters& parameters, const SamplingIdentity& identity, Clock::time_point deadline) {
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Inference deadline expired before admission");
  if (config.config_sha256() != boundary.config_sha256() || tokens.empty() || tokens.size() > 4096)
    return Status::InvalidArgument("Inference boundary config or token count mismatch");
  auto views = boundary.Bind(device, host, static_cast<std::uint32_t>(tokens.size()), stream);
  if (!views.ok()) return views.status();
  views->sampling.parameters = parameters;
  const auto sampling = ValidateSampling(views->sampling); if (!sampling.ok()) return sampling;
  if (!identity.epoch || !identity.plan_seq || !identity.sequence_generation || !identity.sampling_config_id)
    return Status::InvalidArgument("Inference requires nonzero controller sampling identity");
  if (views->embedding.world_size != resources.ffn.world_size() || views->embedding.rank != resources.ffn.rank())
    return Status::InvalidArgument("Inference boundary and backbone rank mismatch");
  const auto safe = resources.weights.ValidateScratch(std::array{device}); if (!safe.ok()) return safe;
  const auto& allocation = resources.workspace.allocation_record();
  const std::array regions{resources.phase_arena, resources.engram_arena, resources.cache_arena,
      resources.ffn_arenas[0], resources.ffn_arenas[1], resources.attention_arena, resources.output_arena,
      resources.compressor_arena, resources.indexer_arenas[0], resources.indexer_arenas[1],
      resources.indexer_arenas[2], resources.indexer_arenas[3], resources.indexer_arenas[4],
      resources.indexer_arenas[5], resources.indexer_arenas[6], resources.indexer_arenas[7],
      EngramDeviceRegion{allocation.address, allocation.bytes}};
  for (const auto region : regions)
    if (!Valid(region) || Overlap(device, region) || Overlap(host, region))
      return Status::InvalidArgument("Inference boundary overlaps backbone storage or storage is invalid");
  try {
    auto operation = std::unique_ptr<InferenceOperation>(new InferenceOperation(config, sequence, resources, *views, identity, deadline));
    auto input = EmbeddingOperation::Start(sequence, views->embedding, resources.weights, hashes, tokens,
        views->input_upload, resources.communicator, operation->resources_.completion, deadline);
    if (!input.ok()) return input.status();
    operation->embedding_.emplace(std::move(*input)); operation->state_ = InferenceState::kEmbedding;
    return operation;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Inference operation allocation failed");
  }
}
Result<InferenceState> InferenceOperation::Advance() {
  try {
    auto result = AdvanceImpl();
    if (!result.ok()) { state_ = InferenceState::kFailed; sequence_.Fail(); }
    return result;
  } catch (const std::bad_alloc&) {
    state_ = InferenceState::kFailed; sequence_.Fail();
    return Status::ResourceExhausted("Inference continuation allocation failed");
  }
}
Result<InferenceState> InferenceOperation::AdvanceImpl() {
  if (state_ == InferenceState::kFailed) return Status::FailedPrecondition("Inference operation failed");
  if (state_ == InferenceState::kComplete) return state_;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Inference step deadline expired");
  if (state_ == InferenceState::kEmbedding) {
    auto ready = embedding_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != EmbeddingState::kComplete) return state_;
    embedding_.reset();
    auto next = BackboneOperation::Start(config_, sequence_, resources_, deadline_); if (!next.ok()) return next.status();
    backbone_ = std::move(*next); state_ = InferenceState::kBackbone; return state_;
  }
  if (state_ == InferenceState::kBackbone) {
    auto ready = backbone_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != BackboneState::kComplete) return state_;
    backbone_.reset();
    auto next = HeadOperation::Start(sequence_, views_.head, resources_.weights, views_.logits,
        resources_.communicator, resources_.completion, deadline_); if (!next.ok()) return next.status();
    head_.emplace(std::move(*next)); state_ = InferenceState::kHead; return state_;
  }
  if (state_ == InferenceState::kHead) {
    auto ready = head_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != HeadOperationState::kComplete) return state_;
    if (views_.embedding.rank + 1 != views_.embedding.world_size) {
      state_ = InferenceState::kComplete; return state_;
    }
    auto next = SamplingOperation::Start(*head_, views_.sampling, identity_, views_.host_candidate,
        resources_.completion, deadline_); if (!next.ok()) return next.status();
    sampling_.emplace(std::move(*next)); state_ = InferenceState::kSampling; return state_;
  }
  auto ready = sampling_->Poll(); if (!ready.ok()) return ready.status();
  if (*ready) state_ = InferenceState::kComplete;
  return state_;
}
Result<EngramDeviceRegion> InferenceOperation::Logits() const {
  if (state_ != InferenceState::kComplete || !head_)
    return Status::FailedPrecondition("Inference logits require completed step");
  return head_->Logits();
}
Result<SamplingObservation> InferenceOperation::Candidate() const {
  if (state_ != InferenceState::kComplete || !sampling_)
    return Status::FailedPrecondition("Inference candidate requires completed last-rank sampling");
  return sampling_->Candidate();
}
Result<RankStepReceipt> InferenceOperation::Receipt() const {
  auto logits = Logits(); if (!logits.ok()) return logits.status();
  RankStepReceipt receipt;
  receipt.identity_ = identity_; receipt.ordinal_ = views_.sampling.parameters.ordinal;
  receipt.processed_ = step_end_; receipt.rank_ = views_.embedding.rank; receipt.world_ = views_.embedding.world_size;
  if (receipt.rank_ + 1 == receipt.world_) {
    auto candidate = Candidate(); if (!candidate.ok()) return candidate.status(); receipt.candidate_ = *candidate;
  }
  return receipt;
}
}  // namespace pih::deepseek_v41
