#include "token_ledger.h"
#include <algorithm>
#include <limits>
#include <new>
#include <type_traits>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<TokenLedger>> TokenLedger::Create(SamplingIdentity identity, std::uint32_t prompt,
    SamplingParameters sampling, TokenStopConfig stopping, std::uint64_t budget, TokenOutputQueue& output) {
  const auto valid = ValidateSamplingParameters(sampling); if (!valid.ok()) return valid;
  if (!identity.epoch || identity.plan_seq || !identity.sequence_generation || !identity.sampling_config_id ||
      !prompt || prompt > 4096 || sampling.ordinal || sampling.suppressed_count ||
      !stopping.maximum || stopping.maximum > 1048576U - prompt)
    return Status::InvalidArgument("Ledger identity, initial sampling or context reservation invalid");
  const auto capacity = stopping.maximum;
  if (capacity > budget / sizeof(AcceptedToken))
    return Status::ResourceExhausted("Ledger record reservation exceeds supplied memory budget");
  auto stop = TokenStopState::Create(std::move(stopping)); if (!stop.ok()) return stop.status();
  try {
    auto ledger = std::unique_ptr<TokenLedger>(new TokenLedger(std::move(*stop)));
    ledger->records_.resize(capacity); // All record storage allocated before execution.
    ledger->identity_ = identity; ledger->sampling_ = sampling; ledger->prompt_ = prompt;
    ledger->output_ = &output;
    return ledger;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Ledger admission allocation failed");
  }
}
Result<TokenSamplingRequest> TokenLedger::Prepare(std::uint64_t plan) {
  if (failed_ || prepared_ || stopping_.finish() != TokenFinish::kNone || count_ >= records_.size())
    return Status::FailedPrecondition("Ledger cannot prepare another sampling decision");
  if (!plan || plan <= last_plan_) return Status::InvalidArgument("Ledger plan sequence must increase, including after abort");
  TokenSamplingRequest request{identity_, sampling_, prompt_ + count_};
  request.identity.plan_seq = plan; request.parameters.ordinal = ordinal_;
  const auto suppression = stopping_.ApplySuppression(request.parameters); if (!suppression.ok()) return suppression;
  auto reserved = output_->Reserve(plan); if (!reserved.ok()) return reserved.status();
  output_lease_ = *reserved;
  prepared_ = request; last_plan_ = plan; return request;
}
Status TokenLedger::MarkInFlight(std::uint64_t plan) {
  if (failed_ || !prepared_ || prepared_->identity.plan_seq != plan || !output_lease_)
    return Status::FailedPrecondition("Ledger has no prepared output reservation");
  return output_->MarkInFlight(*output_lease_);
}
Status TokenLedger::Stage(const SamplingObservation& o, std::string_view bytes) {
  if (failed_ || !prepared_ || staged_) return Status::FailedPrecondition("Ledger has no unstaged prepared decision");
  if (!output_lease_) return Status::FailedPrecondition("Ledger has no output reservation");
  const auto credit = output_->ValidateCommitted(*output_lease_); if (!credit.ok()) return credit;
  const auto& expected = *prepared_; const auto& id = expected.identity;
  if (o.identity.epoch != id.epoch || o.identity.plan_seq != id.plan_seq ||
      o.identity.sequence_generation != id.sequence_generation || o.identity.sampling_config_id != id.sampling_config_id ||
      o.ordinal != expected.parameters.ordinal || o.processed_length != expected.processed_length) {
    failed_ = true; return Status::FailedPrecondition("Sampling observation identity, ordinal or causal position differs");
  }
  const auto valid = ValidateSamplingCandidate(o.candidate, expected.parameters);
  if (!valid.ok()) { failed_ = true; return valid; }
  auto transition = stopping_.Preview(o, bytes);
  if (!transition.ok()) { failed_ = true; return transition.status(); }
  staged_.emplace(std::move(*transition)); return Status::Ok();
}
Result<TokenOutputLease> TokenLedger::CommitLocal(std::uint64_t plan) {
  if (failed_ || !prepared_ || !staged_ || prepared_->identity.plan_seq != plan ||
      count_ >= records_.size() || staged_->accepted_count() != count_ + 1)
    return Status::FailedPrecondition("Ledger cannot commit an absent, stale or incomplete decision");
  if (!output_lease_) return Status::FailedPrecondition("Ledger output reservation is absent");
  const auto credit = output_->ValidateCommitted(*output_lease_); if (!credit.ok()) return credit;
  const bool stochastic = prepared_->parameters.temperature != 0;
  if (stochastic && ordinal_ == std::numeric_limits<std::uint64_t>::max()) {
    failed_ = true; return Status::FailedPrecondition("Sampling ordinal exhausted");
  }
  TokenPublication publication;
  publication.record = {staged_->observation(), staged_->accepted_count(), staged_->finish()};
  const auto bytes = staged_->visible_bytes();
  std::copy(bytes.begin(), bytes.end(), publication.visible.begin()); publication.visible_size = static_cast<unsigned>(bytes.size());
  // Everything that can fail is checked before mutating either local state.
  // The remaining assignments use reserved storage and nonthrowing value types.
  const auto committed = stopping_.Commit(*staged_); if (!committed.ok()) { failed_ = true; return committed; }
  static_assert(std::is_nothrow_copy_assignable_v<AcceptedToken>);
  records_[count_++] = publication.record; processed_ = publication.record.observation.processed_length;
  if (publication.record.finish == TokenFinish::kNone) pending_input_ = publication.record.observation.candidate.token_id;
  else pending_input_.reset();
  if (stochastic) ++ordinal_;
  auto published = output_->Publish(*output_lease_, publication);
  if (!published.ok()) { failed_ = true; return published.status(); } // No rollback after local acceptance.
  staged_.reset(); prepared_.reset(); output_lease_.reset(); return *published;
}
Status TokenLedger::Abort(std::uint64_t plan) {
  if (failed_ || !prepared_ || prepared_->identity.plan_seq != plan)
    return Status::FailedPrecondition("Ledger cannot abort an absent or stale decision");
  if (!output_lease_) return Status::FailedPrecondition("Ledger output reservation is absent");
  const auto aborted = output_->AbortPrepared(*output_lease_); if (!aborted.ok()) return aborted;
  staged_.reset(); prepared_.reset(); output_lease_.reset(); return Status::Ok();
}
Status TokenLedger::AbortRetired(std::uint64_t plan) {
  if (!prepared_ || prepared_->identity.plan_seq != plan || !output_lease_)
    return Status::FailedPrecondition("Ledger cannot retire an absent or stale decision");
  const auto discarded = output_->DiscardRetired(*output_lease_); if (!discarded.ok()) return discarded;
  staged_.reset(); prepared_.reset(); output_lease_.reset(); return Status::Ok();
}
}  // namespace pih::deepseek_v41
