#include "packed_inference_operation.h"

namespace pih::deepseek_v41 {
PackedInferenceOperation::~PackedInferenceOperation() {
  if (state_ != InferenceState::kFailed && state_ != InferenceState::kComplete) Fail();
}
void PackedInferenceOperation::Fail() noexcept {
  state_ = InferenceState::kFailed;
  for (const auto& member : plan_.spans()) if (member.cache_slot < slots_.size() && slots_[member.cache_slot])
    slots_[member.cache_slot]->Fail();
}
Result<std::unique_ptr<PackedInferenceOperation>> PackedInferenceOperation::Start(const FlashConfig& config,
    PackedRequest plan, std::span<BlockSequence* const> sequences, std::span<EngramHashState* const> hashes,
    const PackedBoundaryPlan& boundary, EngramDeviceRegion device, EngramDeviceRegion host,
    std::uintptr_t stream, const PackedBackboneResources& resources, Clock::time_point deadline) {
  if (resources.members.size() != plan.spans().size() || resources.members.empty() || Clock::now() >= deadline)
    return Status::InvalidArgument("Packed inference resource count or deadline invalid");
  auto views = boundary.Bind(device, host, plan.tokens().size(), plan.spans().size(), stream);
  if (!views.ok()) return views.status();
  const auto& first = resources.members.front();
  if (views->ordinary.embedding.world_size != resources.ffn.world_size() ||
      views->ordinary.embedding.rank != resources.ffn.rank())
    return Status::InvalidArgument("Packed inference boundary/backbone rank mismatch");
  const auto protected_boundary = first.weights.ValidateScratch(std::array{device});
  if (!protected_boundary.ok()) return protected_boundary;
  try {
    auto operation = std::unique_ptr<PackedInferenceOperation>(new PackedInferenceOperation(config, std::move(plan), resources, *views, deadline));
    operation->slots_.assign(sequences.begin(), sequences.end());
    operation->member_resources_.reserve(resources.members.size());
    for (const auto& member : resources.members) operation->member_resources_.push_back(member);
    for (auto& member : operation->member_resources_) {
      member.host_counts = views->ordinary.host_counts;
      member.completion.host_error_flag = views->ordinary.host_error;
    }
    operation->resources_.members = operation->member_resources_;
    operation->observations_.reserve(operation->plan_.spans().size());
    auto& owner = operation->member_resources_.front();
    auto embedding = EmbeddingOperation::StartPacked(operation->plan_, sequences, hashes, views->ordinary.embedding,
        owner.weights, views->ordinary.input_upload, owner.communicator, owner.completion, deadline);
    if (!embedding.ok()) return embedding.status();
    operation->embedding_.emplace(std::move(*embedding)); operation->state_ = InferenceState::kEmbedding;
    return operation;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Packed inference admission allocation failed"); }
}
Result<InferenceState> PackedInferenceOperation::Advance() {
  try { auto result = AdvanceImpl(); if (!result.ok()) Fail(); return result; }
  catch (const std::bad_alloc&) { Fail(); return Status::ResourceExhausted("Packed inference continuation allocation failed"); }
}
Result<InferenceState> PackedInferenceOperation::AdvanceImpl() {
  if (state_ == InferenceState::kFailed) return Status::FailedPrecondition("Packed inference failed");
  if (state_ == InferenceState::kComplete) return state_;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Packed inference execution deadline elapsed");
  auto& owner = member_resources_.front();
  if (state_ == InferenceState::kEmbedding) {
    auto ready = embedding_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != EmbeddingState::kComplete) return state_;
    embedding_.reset();
    auto backbone = PackedBackboneOperation::Start(config_, plan_, slots_, resources_, deadline_);
    if (!backbone.ok()) return backbone.status();
    backbone_ = std::move(*backbone); state_ = InferenceState::kBackbone; return state_;
  }
  if (state_ == InferenceState::kBackbone) {
    auto ready = backbone_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != BackboneState::kComplete) return state_;
    auto first = backbone_->Output(0); if (!first.ok()) return first.status();
    auto launch = views_.ordinary.head;
    launch.input.collapse.residual = {first->residual.address, plan_.tokens().size() * 4ULL * 5120 * 2};
    launch.input.collapse.pre = {first->pre.address, plan_.tokens().size() * 4ULL * 4};
    const auto rows = UploadPackedHeadRows(plan_, views_.host_head_rows, views_.device_head_rows, launch.projection.stream);
    if (!rows.ok()) return rows;
    auto head = HeadOperation::StartPacked(plan_, slots_, launch, owner.weights, views_.gathered_logits,
        views_.logits, owner.communicator, owner.completion, deadline_);
    if (!head.ok()) return head.status();
    head_.emplace(std::move(*head)); backbone_.reset(); state_ = InferenceState::kHead; return state_;
  }
  if (state_ == InferenceState::kHead) {
    auto ready = head_->Advance(); if (!ready.ok()) return ready.status();
    if (*ready != HeadOperationState::kComplete) return state_;
    if (views_.ordinary.embedding.rank + 1 != views_.ordinary.embedding.world_size) {
      state_ = InferenceState::kComplete; return state_;
    }
    auto started = StartSampling(); if (!started.ok()) return started;
    state_ = InferenceState::kSampling; return state_;
  }
  auto ready = sampling_->Poll(); if (!ready.ok()) return ready.status();
  if (!*ready) return state_;
  auto observation = sampling_->Candidate(); if (!observation.ok()) return observation.status();
  observations_.push_back(*observation); sampling_.reset();
  if (++sampling_member_ == plan_.spans().size()) { state_ = InferenceState::kComplete; return state_; }
  auto next = StartSampling(); if (!next.ok()) return next; return state_;
}
Status PackedInferenceOperation::StartSampling() {
  auto launch = views_.ordinary.sampling; launch.parameters = plan_.parameters()[sampling_member_];
  auto& owner = member_resources_.front();
  auto operation = SamplingOperation::StartPacked(*head_, sampling_member_, launch, plan_.spans()[sampling_member_].identity,
      views_.ordinary.host_candidate, owner.completion, deadline_);
  if (!operation.ok()) return operation.status(); sampling_.emplace(std::move(*operation)); return Status::Ok();
}
Result<EngramDeviceRegion> PackedInferenceOperation::Logits(std::uint32_t member) const {
  if (state_ != InferenceState::kComplete || !head_) return Status::FailedPrecondition("Packed inference logits incomplete");
  return head_->PackedLogits(member);
}
Result<RankStepReceipt> PackedInferenceOperation::Receipt(std::uint32_t member) const {
  auto logits = Logits(member); if (!logits.ok()) return logits.status();
  const auto& span = plan_.spans()[member];
  RankStepReceipt receipt; receipt.identity_ = span.identity; receipt.ordinal_ = plan_.parameters()[member].ordinal;
  receipt.processed_ = span.position + span.token_count; receipt.rank_ = views_.ordinary.embedding.rank;
  receipt.world_ = views_.ordinary.embedding.world_size;
  if (receipt.rank_ + 1 == receipt.world_) {
    if (member >= observations_.size()) return Status::FailedPrecondition("Packed inference sampling incomplete");
    receipt.candidate_ = observations_[member];
  }
  return receipt;
}
}
