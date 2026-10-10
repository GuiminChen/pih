#include "packed_rank_worker_loop.h"

namespace pih::deepseek_v41 {
Result<std::unique_ptr<PackedRankWorkerLoop>> PackedRankWorkerLoop::Create(const FlashConfig& config,
    const EngramHashState& hashes, const PackedRankWorkerResources& r, std::uint64_t epoch,
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens, Clock::time_point deadline) {
  if (!epoch || !maximum_sequences || maximum_sequences > 32 || maximum_tokens < maximum_sequences || maximum_tokens > 4096 ||
      hashes.position() || Clock::now() >= deadline || r.memory.state() != InferenceMemoryState::kReady ||
      !r.requests.receiver_ready() || !r.receipts.sender_ready() || !r.communicator || !r.completion_event ||
      r.requests.rank() != r.receipts.rank() || r.requests.world() != r.receipts.world() ||
      r.weights.catalog().rank() != r.requests.rank() || r.weights.catalog().world_size() != r.requests.world() ||
      r.weights.catalog().config_sha256() != config.config_sha256())
    return Status::InvalidArgument("Packed worker rank, policy, hash template or resources invalid");
  auto loop = std::unique_ptr<PackedRankWorkerLoop>(new PackedRankWorkerLoop(config, r));
  loop->epoch_ = epoch; loop->maximum_tokens_ = maximum_tokens; loop->deadline_ = deadline;
  loop->ledger_.emplace(epoch, maximum_sequences);
  loop->slots_.resize(maximum_sequences); loop->hashes_.reserve(maximum_sequences); loop->sequences_.reserve(maximum_sequences);
  for (unsigned i = 0; i < maximum_sequences; ++i) {
    loop->hashes_.push_back(hashes); loop->sequences_.push_back(std::make_unique<BlockSequence>(config));
    loop->sequence_views_.push_back(loop->sequences_.back().get()); loop->hash_views_.push_back(&loop->hashes_.back());
  }
  return loop;
}
void PackedRankWorkerLoop::Fail() noexcept {
  state_ = RankWorkerState::kFailed; for (auto& sequence : sequences_) sequence->Fail(); resources_.memory.Quarantine();
}
bool PackedRankWorkerLoop::idle() const noexcept {
  return ledger_ && CanRetire(state_, *ledger_);
}
Result<RankWorkerState> PackedRankWorkerLoop::Poll() {
  try { auto result = PollImpl(); if (!result.ok()) Fail(); return result; }
  catch (const std::bad_alloc&) { Fail(); return Status::ResourceExhausted("Packed rank continuation allocation failed"); }
}
Result<RankWorkerState> PackedRankWorkerLoop::PollImpl() {
  if (state_ == RankWorkerState::kFailed) return Status::FailedPrecondition("Packed rank worker failed");
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Packed rank epoch deadline elapsed");
  auto& r = resources_;
  if (state_ == RankWorkerState::kReceiving) {
    auto frame = r.requests.PollReceivePacked(deadline_, slots_.size(), maximum_tokens_); if (!frame.ok()) return frame.status();
    if (!*frame) return state_;
    received_.emplace(std::move(**frame)); receipts_.clear(); receipt_member_ = 0;
    const auto plan_id = received_->spans().front().identity.plan_seq;
    if (plan_id <= last_plan_) return Status::FailedPrecondition("Packed rank global plan ordering regressed");
    const auto admitted = ledger_->Admit(*received_); if (!admitted.ok()) return admitted;
    // Validate every member before touching state or issuing a cache reset.
    for (unsigned i = 0; i < received_->spans().size(); ++i) {
      const auto& span = received_->spans()[i]; const auto& p = received_->parameters()[i];
      const auto& slot = slots_[span.cache_slot]; const auto& sequence = *sequences_[span.cache_slot];
      if (span.identity.epoch != epoch_) return Status::FailedPrecondition("Packed worker epoch differs from authenticated bootstrap");
      if (!slot.identity.sequence_generation) {
        if (span.finish || span.position || p.ordinal)
          return Status::FailedPrecondition("Packed new slot requires ordinary prefill and fresh RNG");
      } else if (slot.identity.sequence_generation != span.identity.sequence_generation ||
          slot.identity.sampling_config_id != span.identity.sampling_config_id || slot.ordinal != p.ordinal ||
          sequence.next_position() != span.position || p.temperature != slot.parameters.temperature ||
          p.top_p != slot.parameters.top_p || p.top_k != slot.parameters.top_k || p.seed != slot.parameters.seed ||
          p.logprobs != slot.parameters.logprobs || p.top_count != slot.parameters.top_count)
        return Status::FailedPrecondition("Packed member identity, position or sampling policy changed");
    }
    std::vector<PackedRequestMember> members;
    for (unsigned i = 0; i < received_->spans().size(); ++i) {
      const auto& span = received_->spans()[i]; const auto& p = received_->parameters()[i]; auto& slot = slots_[span.cache_slot];
      if (!slot.identity.sequence_generation) {
        auto reset = r.memory.ResetSlot(span.cache_slot, r.weights, r.prefill_workspace, r.communicator, r.completion_event);
        if (!reset.ok()) return reset;
        hashes_[span.cache_slot].Reset(); sequences_[span.cache_slot] = std::make_unique<BlockSequence>(config_);
        sequence_views_[span.cache_slot] = sequences_[span.cache_slot].get(); slot.identity = span.identity; slot.parameters = p;
      }
      slot.last_plan = plan_id;
      if (span.finish) continue;
      InferenceRequest request; request.count = span.token_count; request.sampling.identity = span.identity;
      request.sampling.parameters = p; request.sampling.processed_length = span.position + span.token_count;
      std::copy_n(received_->tokens().begin() + span.token_offset, span.token_count, request.tokens.begin());
      // A later prompt chunk can start at a nonzero position. Decode workspace
      // only holds one token per sequence, so any multi-token member uses the
      // packed token-capacity workspace, including continuation prefill.
      members.push_back({std::move(request), span.cache_slot});
    }
    last_plan_ = plan_id;
    if (members.empty()) { state_ = RankWorkerState::kExecuting; return state_; }
    auto plan = PackedRequest::Create(members, slots_.size(), maximum_tokens_); if (!plan.ok()) return plan.status();
    active_.emplace(std::move(*plan));
    auto started = r.memory.StartRequest(config_, *active_, sequence_views_, hash_views_, r.weights,
        NeedsPrefillWorkspace(*active_) ? r.prefill_workspace : r.decode_workspace, r.communicator, r.completion_event, deadline_);
    if (!started.ok()) return started;
    state_ = RankWorkerState::kExecuting; return state_;
  }
  if (state_ == RankWorkerState::kExecuting) {
    if (active_) { auto ready = r.memory.Advance(); if (!ready.ok()) return ready.status(); if (!*ready) return state_; }
    unsigned active_member = 0;
    for (unsigned i = 0; i < received_->spans().size(); ++i) {
      const auto& span = received_->spans()[i]; auto& slot = slots_[span.cache_slot];
      if (span.finish) {
        RankStepReceipt terminal; terminal.identity_ = span.identity; terminal.ordinal_ = slot.ordinal;
        terminal.processed_ = span.position; terminal.rank_ = r.receipts.rank(); terminal.world_ = r.receipts.world(); terminal.terminal_ = true;
        receipts_.push_back(terminal); slot = {};
      } else {
        auto receipt = r.memory.Receipt(active_member++); if (!receipt.ok()) return receipt.status(); receipts_.push_back(*receipt);
        if (slot.parameters.temperature != 0) {
          if (slot.ordinal == UINT64_MAX) return Status::ResourceExhausted("Packed sampling ordinal exhausted");
          ++slot.ordinal;
        }
      }
    }
    const auto committed = ledger_->CommitRetired(*received_); if (!committed.ok()) return committed;
    auto queued = r.receipts.Queue(receipts_.front()); if (!queued.ok()) return queued;
    state_ = RankWorkerState::kSending; return state_;
  }
  auto sent = r.receipts.PollSend(deadline_); if (!sent.ok()) return sent.status(); if (!*sent) return state_;
  if (++receipt_member_ < receipts_.size()) {
    auto queued = r.receipts.Queue(receipts_[receipt_member_]); if (!queued.ok()) return queued; return state_;
  }
  received_.reset(); active_.reset(); receipts_.clear(); state_ = RankWorkerState::kReceiving; return state_;
}
}
