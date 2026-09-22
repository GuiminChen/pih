#include "supervisor_operation.h"
#include <algorithm>
#include <new>

namespace pih::deepseek_v41 {
Status SupervisorOperation::Start(const WorkerExecutable& executable, const WorkerExecutable& helper,
    const WorkerEnvironment& environment, SupervisorRequest& request,
    std::span<const WorkerBootstrap> placements, std::span<WorkerCgroup* const> groups,
    WorkerCgroup& helper_group, std::uint64_t first_plan, std::chrono::milliseconds grace) {
  if (state_ != SupervisorOperationState::kEmpty || (placements.size() != 2 && placements.size() != 4 && placements.size() != 8) ||
      groups.size() != placements.size() || !first_plan || grace.count() < 0)
    return Status::InvalidArgument("Supervisor operation topology, plan or state invalid");
  const auto& base = placements[0];
  if (!base.startup_ns || base.startup_ns >= base.sequence_ns || base.sequence_ns > INT64_MAX ||
      !base.retirement_ms || base.retirement_ms > 300000 || grace >= std::chrono::milliseconds(base.retirement_ms))
    return Status::InvalidArgument("Supervisor operation deadlines invalid");
  const auto deadline = [](std::uint64_t ns) { return Clock::time_point(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(ns))); };
  const auto startup = deadline(base.startup_ns), lifetime = deadline(base.sequence_ns);
  if (Clock::now() >= startup) return Status::DeadlineExceeded("Supervisor startup expired before launch");
  auto valid = executable.Descriptor(); if (!valid.ok()) return valid.status();
  valid = helper.Descriptor(); if (!valid.ok()) return valid.status();
  for (unsigned i = 0; i < groups.size(); ++i) {
    if (!groups[i] || groups[i] == &helper_group) return Status::InvalidArgument("Supervisor cgroup owner absent or shared");
    for (unsigned j = 0; j < i; ++j) if (groups[i] == groups[j]) return Status::InvalidArgument("Supervisor cgroup owner repeats");
    if (placements[i].rank != i || placements[i].world != groups.size() ||
        placements[i].startup_ns != base.startup_ns || placements[i].sequence_ns != base.sequence_ns ||
        placements[i].retirement_ms != base.retirement_ms ||
        std::any_of(placements[i].nccl_id.begin(), placements[i].nccl_id.end(), [](auto b) { return b != std::byte{}; }))
      return Status::InvalidArgument("Supervisor placements must be ordered with fresh broker IDs and common deadlines");
    auto ready = groups[i]->ReadyDescriptor(); if (!ready.ok()) return ready.status();
    auto empty = groups[i]->Empty(); if (!empty.ok()) return empty.status();
    if (!*empty) return Status::FailedPrecondition("Supervisor rank cgroup populated before launch");
  }
  auto ready = helper_group.ReadyDescriptor(); if (!ready.ok()) return ready.status();
  auto empty = helper_group.Empty(); if (!empty.ok()) return empty.status();
  if (!*empty) return Status::FailedPrecondition("Supervisor helper cgroup populated before launch");
  try {
    placements_.assign(placements.begin(), placements.end());
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Supervisor placement admission allocation failed"); }
  executable_ = &executable; environment_ = &environment; request_ = &request; helper_group_ = &helper_group;
  world_ = static_cast<std::uint32_t>(groups.size()); std::copy(groups.begin(), groups.end(), groups_.begin());
  first_plan_ = first_plan; grace_ = grace; timeout_ = std::chrono::milliseconds(base.retirement_ms);
  state_ = SupervisorOperationState::kStartingBroker;
  const auto started = broker_.Start(helper, helper_group, environment, startup, lifetime);
  if (!started.ok()) { Fail(started); return started; }
  return Status::Ok();
}
Status SupervisorOperation::Fail(Status reason) {
  failure_ = reason; request_->ledger().Fail();
  if (session_) {
    const auto cancelled = session_->Cancel(reason);
    state_ = SupervisorOperationState::kSession;
    if (!cancelled.ok()) { cleanup_failure_ = cancelled; state_ = SupervisorOperationState::kFailedUnreconciled; }
    return cancelled;
  }
  retirement_ = Clock::now() + timeout_; state_ = SupervisorOperationState::kRetiringStartup;
  for (auto& placement : placements_) {
    volatile std::byte* bytes = placement.nccl_id.data();
    for (std::size_t i = 0; i < placement.nccl_id.size(); ++i) bytes[i] = std::byte{};
  }
  return Status::Ok();
}
Status SupervisorOperation::Cancel(Status reason) {
  if (reason.ok() || state_ == SupervisorOperationState::kEmpty || state_ == SupervisorOperationState::kComplete)
    return Status::InvalidArgument("Supervisor cancellation requires an active operation and failure reason");
  if (state_ == SupervisorOperationState::kRetiringStartup || state_ == SupervisorOperationState::kFailedRetired ||
      state_ == SupervisorOperationState::kFailedUnreconciled) return Status::Ok();
  return Fail(std::move(reason));
}
Result<bool> SupervisorOperation::RetireUnlaunched() {
  if (Clock::now() >= retirement_) return Status::DeadlineExceeded("Unlaunched supervisor cleanup expired");
  bool complete = true; Status first = Status::Ok();
  if (broker_.state() != NcclBrokerState::kEmpty && broker_.state() != NcclBrokerState::kRetired) {
    if (broker_.state() != NcclBrokerState::kRetiring) {
      const auto begin = broker_.BeginRetirement(retirement_); if (!begin.ok()) first = begin;
    }
    const auto retired = broker_.PollRetirement();
    if (!retired.ok()) { if (first.ok()) first = retired.status(); complete = false; }
    else if (!*retired) complete = false;
  }
  // No rank process has been launched in this branch. Never kill a newly
  // observed foreign occupant of an otherwise untouched cgroup.
  for (unsigned i = 0; i <= world_; ++i) {
    auto* group = i == world_ ? helper_group_ : groups_[i];
    if (i == world_ && broker_.state() != NcclBrokerState::kEmpty) continue;
    if (group->removed()) continue;
    auto empty = group->Empty();
    if (!empty.ok()) { if (first.ok()) first = empty.status(); continue; }
    if (!*empty) { complete = false; continue; }
    const auto removed = group->Remove(); if (!removed.ok() && first.ok()) first = removed;
  }
  if (!first.ok()) return first;
  return complete;
}
Result<SupervisorOperationState> SupervisorOperation::Poll() {
  try { return PollImpl(); }
  catch (const std::bad_alloc&) {
    const auto error = Status::ResourceExhausted("Supervisor continuation allocation failed");
    const auto retired = Fail(error); if (!retired.ok()) return retired;
    return state_;
  }
}
Result<SupervisorOperationState> SupervisorOperation::PollImpl() {
  const auto failed = [&](Status error) -> Result<SupervisorOperationState> { const auto retired = Fail(error); if (!retired.ok()) return retired; return state_; };
  if (state_ == SupervisorOperationState::kStartingBroker) {
    auto ready = broker_.Poll(); if (!ready.ok()) return failed(ready.status());
    if (!*ready) return state_;
    auto id = broker_.BorrowId(); if (!id.ok()) return failed(id.status());
    for (auto& placement : placements_) {
      placement.nccl_id = *id;
      auto bound = request_->BindBootstrap(placement); if (!bound.ok()) return failed(bound.status());
      placement = std::move(*bound);
    }
    const auto started = group_.Start(*executable_, {groups_.data(), world_}, placements_, *environment_, broker_);
    if (!started.ok()) return failed(started);
    for (auto& placement : placements_) placement.nccl_id.fill(std::byte{});
    state_ = SupervisorOperationState::kConnectingRanks;
  } else if (state_ == SupervisorOperationState::kConnectingRanks) {
    auto ready = group_.Poll(); if (!ready.ok()) return failed(ready.status());
    if (!*ready) return state_;
    auto session = group_.StartGeneration(request_->ledger(), request_->prompt(), request_->token_bytes(), first_plan_, grace_);
    if (!session.ok()) return failed(session.status());
    session_ = std::move(*session); state_ = SupervisorOperationState::kSession;
  } else if (state_ == SupervisorOperationState::kSession || state_ == SupervisorOperationState::kOutputReady) {
    auto result = session_->Poll();
    if (!result.ok()) { cleanup_failure_ = result.status(); state_ = SupervisorOperationState::kFailedUnreconciled; return state_; }
    if (*result == GenerationSessionState::kComplete) state_ = SupervisorOperationState::kComplete;
    else if (*result == GenerationSessionState::kFailedRetired) state_ = SupervisorOperationState::kFailedRetired;
    else if (*result == GenerationSessionState::kFailedUnreconciled) state_ = SupervisorOperationState::kFailedUnreconciled;
    else if (*result == GenerationSessionState::kOutputReady) state_ = SupervisorOperationState::kOutputReady;
    else state_ = SupervisorOperationState::kSession;
    if (!session_->failure().ok()) failure_ = session_->failure();
    if (!session_->retirement_failure().ok()) cleanup_failure_ = session_->retirement_failure();
  } else if (state_ == SupervisorOperationState::kRetiringStartup) {
    Result<bool> retired = false;
    if (!group_.has_startup_custody()) retired = RetireUnlaunched();
    else {
      if (group_.state() != WorkerGroupState::kRetiring && group_.state() != WorkerGroupState::kRetired) {
        const auto begin = group_.BeginStartupRetirement(retirement_);
        if (!begin.ok()) { cleanup_failure_ = begin; state_ = SupervisorOperationState::kFailedUnreconciled; return state_; }
      }
      retired = group_.PollStartupRetirement();
    }
    if (!retired.ok()) { cleanup_failure_ = retired.status(); state_ = SupervisorOperationState::kFailedUnreconciled; }
    else if (*retired) state_ = SupervisorOperationState::kFailedRetired;
  }
  return state_;
}
Result<TokenOutputLease> SupervisorOperation::TakeOutput() {
  if (!session_) return Status::FailedPrecondition("Supervisor has no generation session output");
  auto lease = session_->TakeOutput();
  if (lease.ok() && state_ == SupervisorOperationState::kOutputReady) state_ = SupervisorOperationState::kSession;
  return lease;
}
}  // namespace pih::deepseek_v41
