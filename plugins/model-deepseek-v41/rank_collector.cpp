#include "rank_collector.h"
#include <new>

namespace pih::deepseek_v41 {
RankReceiptCollector::~RankReceiptCollector() { if (active_ && !committed_) Fail(); }
void RankReceiptCollector::Fail() noexcept {
  failed_ = true;
  if (gate_) gate_->Fail();
  if (ledger_) ledger_->Fail();
}
Result<std::unique_ptr<RankReceiptCollector>> RankReceiptCollector::Begin(TokenLedger& ledger, std::uint64_t plan,
    std::span<RankReceiptChannel* const> channels, Clock::time_point deadline) {
  if ((channels.size() != 2 && channels.size() != 4 && channels.size() != 8) || Clock::now() >= deadline)
    return Status::InvalidArgument("Collector world or deadline invalid");
  for (std::size_t i = 0; i < channels.size(); ++i)
    if (!channels[i] || !channels[i]->receiver_ready() || channels[i]->rank() != i || channels[i]->world() != channels.size())
      return Status::InvalidArgument("Collector requires one ready receiver per rank, in rank order");
  try {
    auto collector = std::unique_ptr<RankReceiptCollector>(new RankReceiptCollector);
    auto request = ledger.Prepare(plan); if (!request.ok()) return request.status();
    auto gate = RankCommitGate::Create(*request, static_cast<std::uint32_t>(channels.size()), deadline);
    if (!gate.ok()) {
      const auto aborted = ledger.Abort(plan); if (!aborted.ok()) { ledger.Fail(); return aborted; }
      return gate.status();
    }
    collector->gate_.emplace(std::move(*gate));
    collector->ledger_ = &ledger; collector->request_ = *request; collector->deadline_ = deadline;
    collector->world_ = static_cast<std::uint32_t>(channels.size());
    for (std::size_t i = 0; i < channels.size(); ++i) collector->channels_[i] = channels[i];
    // Only now is worker dispatch permitted. From this boundary onward failure
    // retains the output reservation until controller fault retirement.
    collector->active_ = true;
    const auto flight = ledger.MarkInFlight(plan); if (!flight.ok()) return flight;
    return collector;
  } catch (const std::bad_alloc&) {
    ledger.Fail(); return Status::ResourceExhausted("Rank collector allocation failed");
  }
}
Result<bool> RankReceiptCollector::Poll() {
  if (!active_ || failed_ || committed_) return Status::FailedPrecondition("Rank collector is closed");
  if (Clock::now() >= deadline_) { Fail(); return Status::DeadlineExceeded("Rank collector deadline expired"); }
  for (std::uint32_t rank = 0; rank < world_; ++rank) {
    if (mask_ & (1U << rank)) continue;
    auto received = channels_[rank]->PollReceive(*gate_, deadline_);
    if (!received.ok()) { Fail(); return received.status(); }
    if (*received) mask_ |= 1U << rank;
  }
  return gate_->ready();
}
Result<SamplingObservation> RankReceiptCollector::Candidate() const {
  if (!active_ || failed_ || committed_) return Status::FailedPrecondition("Rank collector is closed");
  return gate_->Candidate();
}
Result<TokenOutputLease> RankReceiptCollector::Commit(std::string_view bytes, RankProcessWatch& processes) {
  if (!active_ || failed_ || committed_ || !gate_->ready())
    return Status::FailedPrecondition("Rank collector cannot commit an incomplete step");
  for (std::uint32_t rank = 0; rank < world_; ++rank) {
    if (processes.world() != world_ || processes.pid(rank) != channels_[rank]->peer_pid()) {
      Fail(); return Status::FailedPrecondition("Commit process watch differs from authenticated channel peers");
    }
    const auto quiet = channels_[rank]->CheckQuiet(*gate_);
    if (!quiet.ok()) { Fail(); return quiet; }
  }
  const auto live = processes.CheckLive(); if (!live.ok()) { Fail(); return live; }
  auto committed = gate_->Commit(*ledger_, bytes);
  if (!committed.ok()) { Fail(); return committed.status(); }
  committed_ = true; active_ = false; return *committed;
}
}  // namespace pih::deepseek_v41
