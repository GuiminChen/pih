#include "embedding_operation.h"

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
EmbeddingOperation::EmbeddingOperation(EmbeddingOperation&& other) noexcept
    : sequence_(other.sequence_), launch_(other.launch_), upload_(other.upload_), communicator_(other.communicator_), resources_(other.resources_),
      deadline_(other.deadline_), reduction_(std::move(other.reduction_)), completion_(std::move(other.completion_)), state_(other.state_) {
  other.sequence_ = nullptr; other.state_ = EmbeddingState::kFailed;
}
EmbeddingOperation::~EmbeddingOperation() { if (sequence_ && state_ != EmbeddingState::kComplete) sequence_->Fail(); }
Result<EmbeddingOperation> EmbeddingOperation::Start(BlockSequence& sequence, TokenEmbeddingLaunch x, const BackboneWeightUpload& weights,
    EngramHashState& hashes, std::span<const std::uint32_t> tokens, const TokenInputUploadLaunch& upload,
    std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline) {
  auto arena = weights.Arena(); if (!arena.ok()) return arena.status();
  if (sequence.config_.config_sha256() != weights.catalog().config_sha256() ||
      (sequence.weight_arena_.bytes && !Same(sequence.weight_arena_, *arena)))
    return Status::InvalidArgument("Embedding weights differ from sequence config or arena");
  auto bound = BindUploadedEmbedding(weights, x); if (!bound.ok()) return bound.status();
  x = *bound;
  const auto protected_upload = weights.ValidateScratch(std::array{upload.device_ids, upload.device_hashes[0], upload.device_hashes[1]});
  if (!protected_upload.ok()) return protected_upload;
  if (tokens.size() != x.tokens || hashes.position() != sequence.start_ ||
      (sequence.hash_state_ && sequence.hash_state_ != &hashes) || !Same(upload.device_ids, x.ids) || upload.stream != x.stream)
    return Status::InvalidArgument("Embedding IDs, hash sequence or upload stream differs");
  const auto upload_valid = ValidateTokenInputUpload(upload, x.tokens); if (!upload_valid.ok()) return upload_valid;
  if (sequence.failed_ || sequence.active_ || sequence.layer_ || sequence.input_ready_ ||
      sequence.config_.config_sha256() == Sha256Digest{} || x.world_size == 1 ||
      sequence.start_ >= FlashConfig::kMaximumPositions || x.tokens > FlashConfig::kMaximumPositions - sequence.start_ ||
      (sequence.start_ && (x.tokens != 1 || sequence.head_end_ != sequence.start_ ||
          (x.rank + 1 == x.world_size && sequence.sample_end_ != sequence.start_))))
    return Status::FailedPrecondition("Sequence is not ready for this ordinary token embedding step");
  if (sequence.stream_ && (sequence.stream_ != x.stream || sequence.communicator_ != communicator ||
      sequence.world_ != x.world_size || sequence.rank_ != x.rank || !Same(sequence.error_, x.error_flag) ||
      !Same(sequence.embedding_weight_, x.weight)))
    return Status::InvalidArgument("Embedding sequence stream, rank, error or weight allocation changed");
  for (const auto retained : sequence.Retained(true))
    for (const auto write : {x.hidden, x.residual, x.pre, x.error_flag}) if (Overlap(write, retained))
      return Status::InvalidArgument("Embedding overwrites a retained sequence input, weight or cache");
  for (const auto retained : sequence.Retained(true, false))
    for (const auto write : {upload.device_ids, upload.device_hashes[0], upload.device_hashes[1]}) if (Overlap(write, retained))
      return Status::InvalidArgument("Input upload overwrites a retained sequence weight or cache");
  for (const auto hash : upload.device_hashes)
    for (const auto region : {x.ids, x.weight, x.hidden, x.residual, x.pre, x.error_flag}) if (Overlap(hash, region))
      return Status::InvalidArgument("Engram hash upload overlaps embedding storage");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Embedding deadline expired before transport admission");
  const auto transport = ValidateBf16Reduction({x.hidden, x.stream, x.world_size, x.rank}, communicator);
  if (!transport.ok()) return transport;
  const auto ready = ValidateEngramCompletionResources(resources); if (!ready.ok()) return ready;
  for (const auto host : {upload.host_ids, upload.host_hashes[0], upload.host_hashes[1]})
    if (Overlap(host, resources.host_error_flag)) return Status::InvalidArgument("Input upload aliases completion host storage");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Embedding deadline expired during admission");
  EmbeddingOperation operation;
  operation.sequence_ = &sequence; operation.launch_ = x; operation.upload_ = upload; operation.communicator_ = communicator;
  operation.resources_ = resources; operation.deadline_ = deadline;
  sequence.active_ = true;
  sequence.weight_arena_ = *arena;
  sequence.stream_ = x.stream; sequence.communicator_ = communicator; sequence.world_ = x.world_size;
  sequence.rank_ = x.rank; sequence.error_ = x.error_flag;
  sequence.hash_state_ = &hashes;
  const auto uploaded = UploadTokenInputs(hashes, sequence.start_, tokens, upload); if (!uploaded.ok()) return uploaded;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Embedding deadline expired after token upload");
  const auto lookup = LaunchTokenEmbeddingLookup(x); if (!lookup.ok()) return lookup;
  auto reduction = EngramReduction::SubmitBf16({x.hidden, x.stream, x.world_size, x.rank}, communicator);
  if (!reduction.ok()) return reduction.status();
  operation.reduction_.emplace(std::move(*reduction)); operation.state_ = EmbeddingState::kWaitingReduction;
  return operation;
}
Result<EmbeddingState> EmbeddingOperation::Advance() {
  auto result = AdvanceImpl(); if (!result.ok() && sequence_) sequence_->Fail(); return result;
}
Result<EmbeddingState> EmbeddingOperation::AdvanceImpl() {
  if (state_ == EmbeddingState::kFailed || !sequence_) return Status::FailedPrecondition("Embedding failed or moved from");
  if (state_ == EmbeddingState::kComplete) return state_;
  const auto previous = state_; state_ = EmbeddingState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Embedding deadline expired; retire generation");
  const auto reduced = reduction_->PollEnqueued(); if (!reduced.ok()) return reduced.status();
  if (*reduced == EngramReductionState::kPending) {
    if (previous != EmbeddingState::kWaitingReduction) return Status::FailedPrecondition("Embedding communicator reused before completion");
    state_ = previous; return state_;
  }
  if (previous == EmbeddingState::kWaitingReduction) {
    const auto transport = ValidateBf16Reduction({launch_.hidden, launch_.stream, launch_.world_size, launch_.rank}, communicator_);
    if (!transport.ok()) return transport;
    if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Embedding deadline expired before mHC initialization");
    const auto expanded = LaunchTokenEmbeddingExpand(launch_); if (!expanded.ok()) return expanded;
    auto completion = EngramCompletion::RecordFlag(launch_.error_flag, launch_.stream, resources_);
    if (!completion.ok()) return completion.status();
    completion_.emplace(std::move(*completion)); state_ = EmbeddingState::kWaitingCompletion; return state_;
  }
  const auto ready = completion_->Poll(); if (!ready.ok()) return ready.status();
  if (!*ready) { state_ = previous; return state_; }
  const auto final = reduction_->PollEnqueued(); if (!final.ok()) return final.status();
  if (*final != EngramReductionState::kEnqueued) return Status::FailedPrecondition("Embedding communicator changed at completion");
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Embedding deadline expired at completion");
  sequence_->residual_ = launch_.residual; sequence_->pre_ = launch_.pre;
  sequence_->embedding_weight_ = launch_.weight; sequence_->embedding_ids_ = launch_.ids;
  sequence_->engram_ids_ = upload_.device_hashes;
  sequence_->input_tokens_ = launch_.tokens; sequence_->input_ready_ = true; sequence_->active_ = false;
  state_ = EmbeddingState::kComplete; return state_;
}
}  // namespace pih::deepseek_v41
