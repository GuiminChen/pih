#include "worker_group.h"
#include "packed_generation_session.h"
#include <algorithm>
#include <limits>
// The CPU transport contract test supplies the process/channel boundaries on
// non-Linux hosts; production sessions still require the Linux implementation.
#if defined(__linux__) || defined(PIH_V41_PORTABLE_SESSION_TEST)
#include "supervisor_request.h"
#endif
namespace pih::deepseek_v41 {
#if defined(__linux__) || defined(PIH_V41_PORTABLE_SESSION_TEST)
Result<std::unique_ptr<PackedGenerationSession>> WorkerGroup::StartPackedGeneration(
    std::uint64_t epoch, std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
    std::uint32_t maximum_pending, std::chrono::milliseconds retirement_timeout,
    std::uint64_t first_plan) {
  if (state_ != WorkerGroupState::kReady || packed_handed_off_ || epoch != identity_.epoch ||
      packed_ready_mask_ != (1U << world_) - 1U)
    return Status::FailedPrecondition("Packed generation requires the all-rank ready barrier");
  auto session = PackedGenerationSession::Create({requests_.data(), world_}, {receipts_.data(), world_},
      *processes_, epoch, maximum_sequences, maximum_tokens, maximum_pending,
      retirement_timeout, first_plan);
  if (!session.ok()) return session.status();
  packed_handed_off_ = true;
  return std::move(*session);
}

#endif
Result<PackedRequest> BuildPackedGenerationPlan(std::span<const PackedGenerationInput> inputs,
    std::uint32_t sequences, std::uint32_t tokens) {
  std::vector<PackedRequestMember> members;
  members.reserve(inputs.size());
  for (const auto& input : inputs) {
    PackedRequestMember member;
    member.cache_slot = input.cache_slot;
    member.request.sampling = input.sampling;
    member.request.finish = input.finish;
    if (!input.finish) {
      if (input.first) {
        if (input.prompt.empty() || input.prompt.size() > member.request.tokens.size() ||
            input.sampling.processed_length != input.prompt.size())
          return Status::InvalidArgument("Packed generation prefill frontier invalid");
        member.request.count = static_cast<std::uint32_t>(input.prompt.size());
        std::copy(input.prompt.begin(), input.prompt.end(), member.request.tokens.begin());
      } else {
        if (!input.pending_input) return Status::InvalidArgument("Packed generation decode input missing");
        member.request.count = 1; member.request.tokens[0] = *input.pending_input;
      }
    }
    members.push_back(std::move(member));
  }
  return PackedRequest::Create(members, sequences, tokens);
}
#if defined(__linux__) || defined(PIH_V41_PORTABLE_SESSION_TEST)
struct PackedGenerationSession::Member final {
  SupervisorRequest* request = nullptr;
  Clock::time_point deadline{};
  std::optional<std::uint32_t> slot;
  std::unique_ptr<RankReceiptCollector> collector;
  TokenSamplingRequest sampling{};
  bool started = false, terminal = false, cancelled = false;
  Status completion = Status::Ok();
};
Result<std::unique_ptr<PackedGenerationSession>> PackedGenerationSession::Create(
    std::span<RankRequestChannel* const> requests, std::span<RankReceiptChannel* const> receipts,
    RankProcessWatch& processes, std::uint64_t epoch, std::uint32_t sequences,
    std::uint32_t tokens, std::uint32_t pending, std::chrono::milliseconds timeout,
    std::uint64_t first_plan) {
  if ((requests.size() != 2 && requests.size() != 4 && requests.size() != 8) ||
      requests.size() != receipts.size() || processes.world() != requests.size() || !epoch ||
      !sequences || sequences > 32 || tokens < sequences || tokens > 4096 ||
      !pending || pending > 4096 || !first_plan || first_plan == UINT64_MAX ||
      timeout.count() <= 0 || timeout > std::chrono::minutes(5))
    return Status::InvalidArgument("Packed generation session bounds invalid");
  for (unsigned rank = 0; rank < requests.size(); ++rank) {
    if (!requests[rank] || !receipts[rank] || !requests[rank]->sender_ready() || !receipts[rank]->receiver_ready() ||
        requests[rank]->rank() != rank || receipts[rank]->rank() != rank ||
        requests[rank]->world() != requests.size() || receipts[rank]->world() != requests.size() ||
        requests[rank]->peer_pid() != processes.pid(rank) || receipts[rank]->peer_pid() != processes.pid(rank))
      return Status::InvalidArgument("Packed generation channels differ from supervised ranks");
  }
  const auto live = processes.CheckLive(); if (!live.ok()) return live;
  auto result = std::unique_ptr<PackedGenerationSession>(new PackedGenerationSession);
  result->world_ = static_cast<std::uint32_t>(requests.size()); result->processes_ = &processes;
  std::copy(requests.begin(), requests.end(), result->requests_.begin());
  std::copy(receipts.begin(), receipts.end(), result->receipts_.begin());
  result->epoch_ = epoch; result->maximum_sequences_ = sequences; result->maximum_tokens_ = tokens;
  result->maximum_pending_ = pending; result->timeout_ = timeout; result->next_plan_ = first_plan;
  result->members_.reserve(sequences + pending); result->selected_.reserve(sequences);
  return result;
}
PackedGenerationSession::~PackedGenerationSession() { if (!members_.empty()) Fail(); }
Status PackedGenerationSession::Admit(SupervisorRequest& request, Clock::time_point deadline) {
  if (state_ == State::kFailed) return failure_;
  auto& ledger = request.ledger(); const auto& identity = ledger.identity();
  if (deadline <= Clock::now() || identity.epoch != epoch_ || !identity.sequence_generation ||
      ledger.failed() || !ledger.records().empty() || ledger.pending_output() || ledger.finish() != TokenFinish::kNone ||
      request.prompt().empty() || request.prompt().size() != ledger.prompt_tokens() ||
      request.prompt().size() > maximum_tokens_ || request.token_bytes().size() != FlashConfig::kVocabularySize)
    return Status::InvalidArgument("Packed generation request admission invalid");
  for (const auto& member : members_)
    if (member->request == &request || member->request->ledger().identity().sequence_generation == identity.sequence_generation)
      return Status::InvalidArgument("Packed generation request identity repeats");
  if (members_.size() >= maximum_sequences_ + maximum_pending_)
    return Status::ResourceExhausted("Packed generation pending admission full");
  auto member = std::make_unique<Member>(); member->request = &request; member->deadline = deadline;
  members_.push_back(std::move(member)); return Status::Ok();
}
bool PackedGenerationSession::HasWork() const noexcept { return !members_.empty() && state_ != State::kFailed; }
void PackedGenerationSession::Fail() noexcept {
  state_ = State::kFailed;
  if (failure_.ok()) failure_ = std::move(stopped_status_);
  for (auto& member : members_) {
    member->request->ledger().Fail(); if (member->collector) member->collector->Fail();
  }
}
Status PackedGenerationSession::Cancel(std::uint64_t generation) {
  if (state_ == State::kFailed) return failure_;
  for (auto& member : members_) {
    if (member->request->ledger().identity().sequence_generation != generation) continue;
    member->cancelled = true; member->completion = Status::Unavailable("Packed generation request cancelled");
    // Only controller-owned committed publications may be discarded here.
    // Leases already transferred to callers remain their responsibility.
    for (auto it = outputs_.begin(); it != outputs_.end();) {
      if (it->request != member->request) { ++it; continue; }
      const auto discarded = it->request->output().Release(it->lease);
      if (!discarded.ok()) { failure_ = discarded; Fail(); return discarded; }
      it = outputs_.erase(it);
    }
    return Status::Ok();
  }
  return Status::FailedPrecondition("Packed generation cancellation identity absent");
}
Status PackedGenerationSession::PreparePlan() {
  // Unstarted cancellation has no device state and consumes no cache slot.
  for (auto it = members_.begin(); it != members_.end();) {
    if (!(*it)->cancelled || (*it)->started) { ++it; continue; }
    completed_.push_back({(*it)->request, (*it)->completion}); it = members_.erase(it);
  }
  for (auto& member : members_) {
    if (member->slot) continue;
    for (unsigned slot = 0; slot < maximum_sequences_; ++slot) {
      if (std::none_of(members_.begin(), members_.end(), [slot](const auto& other) { return other->slot == slot; })) {
        member->slot = slot; break;
      }
    }
  }
  std::vector<Member*> ordered;
  for (auto& member : members_) if (member->slot) ordered.push_back(member.get());
  std::sort(ordered.begin(), ordered.end(), [this](const auto* a, const auto* b) {
    // Alternate compute priority so full-budget prompts fit while surviving
    // decodes also progress under a continuous stream of arriving prompts.
    if (a->started != b->started) return prefer_prefill_ ? !a->started : a->started;
    return a->request->ledger().identity().sequence_generation < b->request->ledger().identity().sequence_generation;
  });
  const bool terminals = std::any_of(ordered.begin(), ordered.end(), [](const auto* member) {
    return member->started && (member->cancelled || member->request->ledger().finish() != TokenFinish::kNone);
  });
  if (ordered.empty()) return Status::Ok();
  if (next_plan_ == UINT64_MAX) return Status::ResourceExhausted("Packed generation plan identity exhausted");
  plan_deadline_ = Clock::now() + timeout_;
  std::uint32_t token_count = 0;
  std::vector<PackedGenerationInput> inputs;
  selected_.clear();
  for (auto* member : ordered) {
    auto& ledger = member->request->ledger();
    const bool terminal = member->started && (member->cancelled || ledger.finish() != TokenFinish::kNone);
    if (terminal != terminals) continue;
    const auto count = terminal ? 0U : member->started ? 1U : static_cast<std::uint32_t>(member->request->prompt().size());
    if (count > maximum_tokens_ - token_count) continue;
    if (terminal) {
      member->sampling = {ledger.identity(), ledger.initial_sampling(), ledger.model_processed_length()};
      member->sampling.identity.plan_seq = next_plan_; member->sampling.parameters.ordinal = ledger.ordinal();
    } else {
      auto collector = RankReceiptCollector::Begin(ledger, next_plan_, {receipts_.data(), world_}, plan_deadline_);
      if (!collector.ok()) {
        if (collector.status().code() == StatusCode::kResourceExhausted && !ledger.failed()) continue;
        return collector.status();
      }
      member->collector = std::move(*collector); member->sampling = member->collector->request();
    }
    member->terminal = terminal;
    selected_.push_back(member); token_count += count;
    inputs.push_back({member->sampling, member->request->prompt(), ledger.pending_input(), *member->slot, !member->started, terminal});
  }
  if (selected_.empty()) return Status::Ok();
  const bool next_prefer_prefill = !selected_.front()->started ? false : true;
  auto plan = BuildPackedGenerationPlan(inputs, maximum_sequences_, maximum_tokens_);
  if (!plan.ok()) return plan.status();
  // Scheduling priority may differ from the canonical packed wire order.
  std::sort(selected_.begin(), selected_.end(), [](const auto* a, const auto* b) {
    return a->sampling.identity.sequence_generation < b->sampling.identity.sequence_generation;
  });
  const auto live = processes_->CheckLive(); if (!live.ok()) return live;
  for (unsigned rank = 0; rank < world_; ++rank) {
    const auto queued = requests_[rank]->QueuePacked(*plan); if (!queued.ok()) return queued;
  }
  if (!terminals) prefer_prefill_ = next_prefer_prefill;
  ++next_plan_; sent_mask_ = terminal_mask_ = 0; collect_index_ = 0; state_ = State::kSending;
  return Status::Ok();
}
Status PackedGenerationSession::FinishPlan() {
  // All canonical member frames have been consumed from every rank before
  // Commit performs its quiet boundary checks. A peer's valid frame is never
  // mistaken for unexpected data at another request's commit boundary.
  const auto live = processes_->CheckLive(); if (!live.ok()) return live;
  for (auto* member : selected_) {
    if (member->terminal) continue;
    auto observation = member->collector->Candidate(); if (!observation.ok()) return observation.status();
    if (observation->candidate.token_id >= member->request->token_bytes().size())
      return Status::FailedPrecondition("Packed generation sampled token exceeds vocabulary");
    auto publication = member->collector->Commit(member->request->token_bytes()[observation->candidate.token_id], *processes_);
    if (!publication.ok()) return publication.status();
    member->collector.reset(); member->started = true;
    if (member->cancelled) {
      const auto discarded = member->request->output().Release(*publication); if (!discarded.ok()) return discarded;
    } else outputs_.push_back({member->request, *publication});
  }
  for (auto* member : selected_) {
    if (!member->terminal) continue;
    completed_.push_back({member->request, member->completion});
    const auto found = std::find_if(members_.begin(), members_.end(), [member](const auto& owned) { return owned.get() == member; });
    members_.erase(found);
  }
  selected_.clear(); state_ = State::kIdle;
  return Status::Ok();
}
Status PackedGenerationSession::PollImpl() {
  if (state_ == State::kFailed) return failure_;
  const auto live = processes_->CheckLive(); if (!live.ok()) return live;
  for (auto& member : members_) {
    if (!member->cancelled && Clock::now() >= member->deadline) {
      const auto cancelled = Cancel(member->request->ledger().identity().sequence_generation);
      if (!cancelled.ok()) return cancelled;
      member->completion = Status::DeadlineExceeded("Packed generation request deadline elapsed");
    }
  }
  if (state_ == State::kIdle) return PreparePlan();
  if (Clock::now() >= plan_deadline_) return Status::DeadlineExceeded("Packed generation plan drain expired");
  if (state_ == State::kSending) {
    for (unsigned rank = 0; rank < world_; ++rank) {
      if (sent_mask_ & (1U << rank)) continue;
      auto sent = requests_[rank]->PollSendPacked(plan_deadline_); if (!sent.ok()) return sent.status();
      if (*sent) sent_mask_ |= 1U << rank;
    }
    if (sent_mask_ == (1U << world_) - 1U) state_ = State::kCollecting;
    return Status::Ok();
  }
  while (collect_index_ < selected_.size()) {
    auto* member = selected_[collect_index_];
    if (member->terminal) {
      for (unsigned rank = 0; rank < world_; ++rank) {
        if (terminal_mask_ & (1U << rank)) continue;
        auto received = receipts_[rank]->PollTerminal(member->sampling.identity, member->sampling.parameters.ordinal,
            member->sampling.processed_length, plan_deadline_);
        if (!received.ok()) return received.status();
        if (*received) terminal_mask_ |= 1U << rank;
      }
      if (terminal_mask_ != (1U << world_) - 1U) return Status::Ok();
      terminal_mask_ = 0;
    } else {
      auto ready = member->collector->Poll(); if (!ready.ok()) return ready.status();
      if (!*ready) return Status::Ok();
    }
    ++collect_index_;
  }
  return FinishPlan();
}
Status PackedGenerationSession::Poll() {
  try {
    const auto status = PollImpl(); if (!status.ok()) { failure_ = status; Fail(); } return status;
  } catch (...) {
    failure_ = Status::Internal("Packed generation continuation failed"); Fail(); return failure_;
  }
}
Result<std::optional<PackedGenerationOutput>> PackedGenerationSession::TakeOutput() {
  if (outputs_.empty()) return std::optional<PackedGenerationOutput>{};
  const auto output = outputs_.front(); outputs_.pop_front(); return std::optional<PackedGenerationOutput>{output};
}
Result<std::optional<PackedGenerationCompleted>> PackedGenerationSession::TakeCompleted() {
  if (completed_.empty()) return std::optional<PackedGenerationCompleted>{};
  auto completion = std::move(completed_.front()); completed_.pop_front();
  return std::optional<PackedGenerationCompleted>{std::move(completion)};
}
#endif
}
