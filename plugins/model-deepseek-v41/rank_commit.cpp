#include "rank_commit.h"
#include <bit>

namespace pih::deepseek_v41 {
namespace {
bool Same(const SamplingIdentity& a, const SamplingIdentity& b) {
  return a.epoch == b.epoch && a.plan_seq == b.plan_seq &&
      a.sequence_generation == b.sequence_generation && a.sampling_config_id == b.sampling_config_id;
}
}
std::array<std::uint8_t, 256> RankStepReceipt::Encode() const noexcept {
  std::array<std::uint8_t, 256> out{};
  const auto put = [&](unsigned at, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) out[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
  };
  put(0, 0x31524b4e41524950ULL, 8); // "PIRANKR1"
  put(8, 1, 4); put(12, out.size(), 4);
  put(16, rank_, 4); put(20, world_, 4); put(24, processed_, 4); put(28, terminal_ ? 2 : candidate_ ? 1 : 0, 4);
  put(32, identity_.epoch, 8); put(40, identity_.plan_seq, 8);
  put(48, identity_.sequence_generation, 8); put(56, identity_.sampling_config_id, 8); put(64, ordinal_, 8);
  if (candidate_) {
    const auto& c = candidate_->candidate;
    put(72, c.token_id, 4); put(76, c.rng_word, 4); put(80, c.top_count, 4);
    put(84, c.reserved, 4); put(88, std::bit_cast<std::uint32_t>(c.selected_logprob), 4);
    for (unsigned i = 0; i < 20; ++i) {
      put(92 + i * 4, c.top_ids[i], 4);
      put(172 + i * 4, std::bit_cast<std::uint32_t>(c.top_logprobs[i]), 4);
    }
  }
  return out;
}
Status RankCommitGate::SubmitWire(std::uint32_t source, std::span<const std::uint8_t> frame) {
  if (failed_ || committed_) return Status::FailedPrecondition("Rank commit gate is closed");
  const auto malformed = [&]() {
    failed_ = true; return Status::InvalidArgument("Malformed rank completion frame");
  };
  if (frame.size() != 256) return malformed();
  const auto get = [&](unsigned at, unsigned size) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < size; ++i) value |= std::uint64_t(frame[at + i]) << (8 * i);
    return value;
  };
  if (get(0, 8) != 0x31524b4e41524950ULL || get(8, 4) != 1 || get(12, 4) != 256 ||
      get(28, 4) > 1 || get(252, 4)) return malformed();
  RankStepReceipt receipt;
  receipt.rank_ = static_cast<std::uint32_t>(get(16, 4)); receipt.world_ = static_cast<std::uint32_t>(get(20, 4));
  receipt.processed_ = static_cast<std::uint32_t>(get(24, 4));
  receipt.identity_ = {get(32, 8), get(40, 8), get(48, 8), get(56, 8)}; receipt.ordinal_ = get(64, 8);
  if (get(28, 4)) {
    SamplingObservation observation{receipt.identity_, receipt.ordinal_, receipt.processed_, {}};
    auto& c = observation.candidate;
    c.token_id = static_cast<std::uint32_t>(get(72, 4)); c.rng_word = static_cast<std::uint32_t>(get(76, 4));
    c.top_count = static_cast<std::uint32_t>(get(80, 4)); c.reserved = static_cast<std::uint32_t>(get(84, 4));
    c.selected_logprob = std::bit_cast<float>(static_cast<std::uint32_t>(get(88, 4)));
    if (c.top_count > 20 || c.reserved) return malformed();
    for (unsigned i = 0; i < 20; ++i) {
      c.top_ids[i] = static_cast<std::uint32_t>(get(92 + i * 4, 4));
      c.top_logprobs[i] = std::bit_cast<float>(static_cast<std::uint32_t>(get(172 + i * 4, 4)));
      if (i >= c.top_count && (get(92 + i * 4, 4) || get(172 + i * 4, 4))) return malformed();
    }
    receipt.candidate_ = observation;
  } else {
    for (unsigned i = 72; i < 252; ++i) if (frame[i]) return malformed();
  }
  return Submit(source, receipt);
}
Result<RankCommitGate> RankCommitGate::Create(const TokenSamplingRequest& request, std::uint32_t world, Clock::time_point deadline) {
  const auto& id = request.identity;
  if (!id.epoch || !id.plan_seq || !id.sequence_generation || !id.sampling_config_id ||
      !request.processed_length || request.processed_length > FlashConfig::kMaximumPositions ||
      (world != 2 && world != 4 && world != 8))
    return Status::InvalidArgument("Rank commit request identity, position or world invalid");
  const auto parameters = ValidateSamplingParameters(request.parameters); if (!parameters.ok()) return parameters;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Rank commit deadline expired");
  RankCommitGate gate; gate.request_ = request; gate.world_ = world; gate.deadline_ = deadline; return gate;
}
Status RankCommitGate::Submit(std::uint32_t source, const RankStepReceipt& receipt) {
  if (failed_ || committed_) return Status::FailedPrecondition("Rank commit gate is closed");
  if (Clock::now() >= deadline_) { failed_ = true; return Status::DeadlineExceeded("Rank completion deadline expired"); }
  if (receipt.terminal() || source >= world_ || receipt.rank() != source || receipt.world() != world_ ||
      (mask_ & (1U << source)) || !Same(receipt.identity(), request_.identity) ||
      receipt.ordinal() != request_.parameters.ordinal || receipt.processed_length() != request_.processed_length ||
      bool(receipt.candidate()) != (source + 1 == world_)) {
    failed_ = true; return Status::FailedPrecondition("Rank completion duplicate, identity or causal position mismatch");
  }
  if (receipt.candidate()) {
    const auto& candidate = *receipt.candidate();
    if (!Same(candidate.identity, request_.identity) || candidate.ordinal != request_.parameters.ordinal ||
        candidate.processed_length != request_.processed_length) {
      failed_ = true; return Status::FailedPrecondition("Rank sampling observation mismatch");
    }
    const auto valid = ValidateSamplingCandidate(candidate.candidate, request_.parameters);
    if (!valid.ok()) { failed_ = true; return valid; }
    candidate_ = candidate;
  }
  mask_ |= 1U << source; return Status::Ok();
}
Result<TokenOutputLease> RankCommitGate::Commit(TokenLedger& ledger, std::string_view bytes) {
  if (!ready() || !candidate_) return Status::FailedPrecondition("Token commit requires every rank completion");
  if (Clock::now() >= deadline_) { failed_ = true; ledger.Fail(); return Status::DeadlineExceeded("Token commit deadline expired"); }
  // No asynchronous wait may separate the caller's final fault recheck from
  // this serialized local acceptance/publication boundary.
  failed_ = true;
  const auto staged = ledger.Stage(*candidate_, bytes); if (!staged.ok()) { ledger.Fail(); return staged; }
  if (Clock::now() >= deadline_) { ledger.Fail(); return Status::DeadlineExceeded("Token commit deadline expired during staging"); }
  auto committed = ledger.CommitLocal(request_.identity.plan_seq); if (!committed.ok()) { ledger.Fail(); return committed.status(); }
  failed_ = false; committed_ = true; return *committed;
}
Result<SamplingObservation> RankCommitGate::Candidate() const {
  if (!ready() || !candidate_) return Status::FailedPrecondition("Candidate requires every rank completion");
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Rank candidate deadline expired");
  return *candidate_;
}
}  // namespace pih::deepseek_v41
