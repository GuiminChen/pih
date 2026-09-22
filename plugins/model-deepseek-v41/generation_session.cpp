#include "generation_session.h"
#include <algorithm>
#include <new>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<GenerationSession>> GenerationSession::Create(std::unique_ptr<GenerationLoop> generation,
    std::span<RankLifecycleChannel* const> commands, std::span<RankLifecycleChannel* const> notices,
    std::span<WorkerCgroup* const> cgroups, NcclBootstrapBroker& broker,
    GenerationLoop::Clock::time_point startup,
    std::chrono::milliseconds grace, std::chrono::milliseconds timeout) {
  if (!generation || generation->state_ != GenerationState::kPreparing || !generation->processes_ ||
      grace.count() < 0 || timeout.count() <= 0 || grace >= timeout || timeout > std::chrono::minutes(5) ||
      commands.size() != generation->world_ || notices.size() != generation->world_ ||
      cgroups.size() != generation->world_ ||
      startup <= GenerationLoop::Clock::now() || startup >= generation->deadline_)
    return Status::InvalidArgument("Generation session requires a fresh loop and bounded retirement deadlines");
  auto broker_ready = broker.Poll(); if (!broker_ready.ok()) return broker_ready.status();
  if (!*broker_ready || broker.lifetime_deadline() <= startup)
    return Status::FailedPrecondition("Generation session requires a live admitted bootstrap broker");
  for (std::uint32_t rank = 0; rank < generation->world_; ++rank) {
    if (!cgroups[rank]) return Status::InvalidArgument("Generation cgroup owner is absent");
    if (cgroups[rank] == broker.cgroup_owner()) return Status::InvalidArgument("Generation rank shares broker cgroup");
    const auto ready = cgroups[rank]->ReadyDescriptor(); if (!ready.ok()) return ready.status();
    for (std::uint32_t prior = 0; prior < rank; ++prior)
      if (cgroups[rank] == cgroups[prior]) return Status::InvalidArgument("Generation cgroup owner repeats");
    if (!commands[rank] || !notices[rank] || !commands[rank]->sender_ready() || !notices[rank]->receiver_ready() ||
        commands[rank]->rank() != rank || notices[rank]->rank() != rank ||
        commands[rank]->world() != generation->world_ || notices[rank]->world() != generation->world_ ||
        commands[rank]->peer_pid() != generation->processes_->pid(rank) ||
        notices[rank]->peer_pid() != generation->processes_->pid(rank))
      return Status::InvalidArgument("Lifecycle channels do not match supervised worker identities");
  }
  try {
    auto session = std::unique_ptr<GenerationSession>(new GenerationSession);
    session->generation_ = std::move(generation); session->grace_ = grace; session->timeout_ = timeout;
    std::copy(commands.begin(), commands.end(), session->commands_.begin());
    std::copy(notices.begin(), notices.end(), session->notices_.begin());
    std::copy(cgroups.begin(), cgroups.end(), session->cgroups_.begin());
    session->lifecycle_deadline_ = startup;
    session->broker_ = &broker;
    return session;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Generation session allocation failed"); }
}
Status GenerationSession::BeginBrokerRetirement() {
  if (broker_->state() == NcclBrokerState::kRetiring || broker_->state() == NcclBrokerState::kRetired)
    return Status::Ok();
  return broker_->BeginRetirement(lifecycle_deadline_);
}
Status GenerationSession::BeginRetirement(Status reason) {
  generation_->Fail(); failure_ = std::move(reason); state_ = GenerationSessionState::kFailedUnreconciled;
  const auto now = RankProcessRetirement::Clock::now();
  lifecycle_deadline_ = now + timeout_; cgroup_kill_after_ = now + grace_;
  const auto broker = BeginBrokerRetirement();
  if (!broker.ok()) retirement_failure_ = broker; // Still attempt rank retirement.
  auto retirement = RankProcessRetirement::Create(*generation_->processes_, generation_->ledger_, now + grace_, now + timeout_);
  if (!retirement.ok()) { retirement_failure_ = retirement.status(); return retirement_failure_; }
  retirement_ = std::move(*retirement); state_ = GenerationSessionState::kRetiring; return Status::Ok();
}
Status GenerationSession::BeginNormalRetirement() {
  // Called only after GenerationLoop has admitted every terminal receipt.
  const auto live = generation_->processes_->CheckLive(); if (!live.ok()) return live;
  lifecycle_deadline_ = GenerationLoop::Clock::now() + timeout_;
  lifecycle_mask_ = 0;
  for (std::uint32_t rank = 0; rank < generation_->world_; ++rank) {
    const auto queued = commands_[rank]->Queue(RankLifecycleKind::kRetire, generation_->ledger_.identity());
    if (!queued.ok()) return queued;
  }
  state_ = GenerationSessionState::kSendingRetire;
  return Status::Ok();
}
std::uint32_t GenerationSession::removed_cgroups_mask() const noexcept {
  std::uint32_t mask = 0;
  for (unsigned i = 0; i < generation_->world_; ++i) if (cgroups_[i]->removed()) mask |= 1U << i;
  return mask;
}
Status GenerationSession::KillCgroups() {
  Status first = Status::Ok();
  for (unsigned i = 0; i < generation_->world_; ++i) {
    if (cgroups_[i]->removed()) continue;
    const auto killed = cgroups_[i]->Kill(); if (!killed.ok() && first.ok()) first = killed;
  }
  return first;
}
Result<bool> GenerationSession::CleanCgroups() {
  if (GenerationLoop::Clock::now() >= lifecycle_deadline_)
    return Status::DeadlineExceeded("Worker cgroup cleanup deadline expired");
  bool complete = true;
  for (unsigned i = 0; i < generation_->world_; ++i) {
    if (cgroups_[i]->removed()) continue;
    const auto empty = cgroups_[i]->Empty(); if (!empty.ok()) return empty.status();
    if (!*empty) {
      if (failure_.ok()) return Status::FailedPrecondition("Worker descendants remain after normal rank exit");
      const auto killed = cgroups_[i]->Kill(); if (!killed.ok()) return killed;
      complete = false; continue;
    }
    const auto removed = cgroups_[i]->Remove(); if (!removed.ok()) return removed;
  }
  return complete;
}
Status GenerationSession::Cancel(Status reason) {
  if (reason.ok()) return Status::InvalidArgument("Cancellation requires a failure reason");
  // Repeated supervisor cancellation must preserve the completed rank reap and
  // its original deadline while cgroups/broker finish fault cleanup. Normal
  // cleanup still needs to transition to fault retirement on its first cancel.
  if (state_ == GenerationSessionState::kCleaningCgroups && !failure_.ok()) return Status::Ok();
  if (state_ == GenerationSessionState::kRetiring || state_ == GenerationSessionState::kFailedRetired ||
      state_ == GenerationSessionState::kFailedUnreconciled) return Status::Ok();
  if (state_ == GenerationSessionState::kComplete) return Status::FailedPrecondition("Completed generation cannot be cancelled");
  return BeginRetirement(std::move(reason));
}
Result<GenerationSessionState> GenerationSession::Poll() {
  try { return PollImpl(); }
  catch (const std::bad_alloc&) {
    generation_->Fail(); state_ = GenerationSessionState::kFailedUnreconciled;
    failure_ = Status::ResourceExhausted("Generation session continuation allocation failed; retirement unproven");
    return failure_;
  }
}
Result<GenerationSessionState> GenerationSession::PollImpl() {
  if (state_ == GenerationSessionState::kWaitingReady || state_ == GenerationSessionState::kSendingRetire ||
      state_ == GenerationSessionState::kWaitingReleased) {
    const auto fail = [&](Status reason) -> Result<GenerationSessionState> {
      const auto started = BeginRetirement(std::move(reason)); if (!started.ok()) return started;
      return state_;
    };
    if (GenerationLoop::Clock::now() >= lifecycle_deadline_)
      return fail(Status::DeadlineExceeded("All-rank lifecycle barrier expired"));
    if (state_ == GenerationSessionState::kWaitingReady) {
      const auto broker = broker_->Poll();
      if (!broker.ok()) return fail(broker.status());
      if (!*broker) return fail(Status::FailedPrecondition("NCCL bootstrap broker lost readiness"));
      const auto live = generation_->processes_->CheckLive(); if (!live.ok()) return fail(live);
    }
    // Once retire is authorized, released workers may exit normally. Do not
    // apply the all-live predicate here; each missing notice is still bounded
    // by EOF/deadline and exact authenticated frame validation.
    for (std::uint32_t rank = 0; rank < generation_->world_; ++rank) {
      const auto bit = 1U << rank;
      if (lifecycle_mask_ & bit) continue;
      auto ready = state_ == GenerationSessionState::kSendingRetire
          ? commands_[rank]->PollSend(lifecycle_deadline_)
          : notices_[rank]->PollReceive(state_ == GenerationSessionState::kWaitingReady
              ? RankLifecycleKind::kReady : RankLifecycleKind::kReleased,
              generation_->ledger_.identity(), lifecycle_deadline_);
      if (!ready.ok()) return fail(ready.status());
      if (*ready) lifecycle_mask_ |= bit;
    }
    if (lifecycle_mask_ == (1U << generation_->world_) - 1U) {
      lifecycle_mask_ = 0;
      if (state_ == GenerationSessionState::kWaitingReady) {
        const auto live = generation_->processes_->CheckLive(); if (!live.ok()) return fail(live);
        lifecycle_deadline_ = GenerationLoop::Clock::now() + timeout_;
        const auto retiring = BeginBrokerRetirement(); if (!retiring.ok()) return fail(retiring);
        state_ = GenerationSessionState::kReapingBootstrap;
      } else if (state_ == GenerationSessionState::kSendingRetire) state_ = GenerationSessionState::kWaitingReleased;
      else state_ = GenerationSessionState::kReaping;
    }
    return state_;
  }
  if (state_ == GenerationSessionState::kReapingBootstrap) {
    const auto live = generation_->processes_->CheckLive();
    auto retired = broker_->PollRetirement();
    if (!live.ok() || !retired.ok() || GenerationLoop::Clock::now() >= generation_->deadline_) {
      const auto error = !live.ok() ? live : !retired.ok() ? retired.status() : Status::DeadlineExceeded("Generation expired during bootstrap retirement");
      const auto started = BeginRetirement(error); if (!started.ok()) return started;
    } else if (*retired) state_ = GenerationSessionState::kRunning;
    return state_;
  }
  if (state_ == GenerationSessionState::kReaping) {
    const auto reaped = generation_->processes_->PollNormalExit(lifecycle_deadline_);
    if (!reaped.ok()) {
      const auto retired = BeginRetirement(reaped.status()); if (!retired.ok()) return retired;
    } else if (*reaped) state_ = GenerationSessionState::kCleaningCgroups;
    return state_;
  }
  if (state_ == GenerationSessionState::kCleaningCgroups) {
    const auto clean = CleanCgroups();
    if (!clean.ok()) {
      if (failure_.ok()) { const auto retired = BeginRetirement(clean.status()); if (!retired.ok()) return retired; }
      else { retirement_failure_ = clean.status(); state_ = GenerationSessionState::kFailedUnreconciled; }
    } else if (*clean) {
      auto broker = broker_->PollRetirement();
      if (!broker.ok()) { retirement_failure_ = broker.status(); state_ = GenerationSessionState::kFailedUnreconciled; }
      else if (*broker) state_ = failure_.ok() ? GenerationSessionState::kComplete : GenerationSessionState::kFailedRetired;
    }
    return state_;
  }
  if (state_ == GenerationSessionState::kOutputReady) {
    const auto live = generation_->processes_->CheckLive();
    if (!live.ok() || GenerationLoop::Clock::now() >= generation_->deadline_) {
      const auto retired = BeginRetirement(live.ok() ? Status::DeadlineExceeded("Generation deadline expired with pending output") : live);
      if (!retired.ok()) return retired;
    }
    return state_;
  }
  if (state_ == GenerationSessionState::kRetiring) {
    if (broker_->state() == NcclBrokerState::kRetiring) {
      const auto broker = broker_->PollRetirement();
      if (!broker.ok()) retirement_failure_ = broker.status(); // Do not skip rank kill/reap.
    }
    Status killed = Status::Ok();
    if (GenerationLoop::Clock::now() >= cgroup_kill_after_) killed = KillCgroups();
    auto progress = retirement_->Poll();
    if (!progress.ok()) { retirement_failure_ = progress.status(); state_ = GenerationSessionState::kFailedUnreconciled; return state_; }
    if (*progress == RankRetirementState::kComplete) {
      const auto discarded = retirement_->DiscardOutput();
      if (!discarded.ok()) { retirement_failure_ = discarded; state_ = GenerationSessionState::kFailedUnreconciled; }
      else state_ = GenerationSessionState::kCleaningCgroups;
    }
    if (!killed.ok()) { retirement_failure_ = killed; state_ = GenerationSessionState::kFailedUnreconciled; }
    return state_;
  }
  if (state_ != GenerationSessionState::kRunning) return state_;
  auto progress = generation_->Poll();
  if (!progress.ok()) { const auto retired = BeginRetirement(progress.status()); if (!retired.ok()) return retired; return state_; }
  if (*progress == GenerationState::kOutputReady) {
    auto publication = generation_->TakeOutput();
    if (!publication.ok()) { const auto retired = BeginRetirement(publication.status()); if (!retired.ok()) return retired; return state_; }
    output_ = *publication; state_ = GenerationSessionState::kOutputReady;
  } else if (*progress == GenerationState::kComplete) {
    const auto normal = BeginNormalRetirement();
    if (!normal.ok()) { const auto retired = BeginRetirement(normal); if (!retired.ok()) return retired; }
  }
  return state_;
}
Result<TokenOutputLease> GenerationSession::TakeOutput() {
  if (!output_) return Status::FailedPrecondition("Generation session has no committed output lease");
  const auto output = *output_; output_.reset();
  if (state_ == GenerationSessionState::kOutputReady) state_ = GenerationSessionState::kRunning;
  // Already committed output remains retrievable even during fault retirement.
  return output;
}
}  // namespace pih::deepseek_v41
