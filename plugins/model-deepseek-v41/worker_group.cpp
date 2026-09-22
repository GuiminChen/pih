#include "worker_group.h"
#include <algorithm>
#include <bit>
#include <new>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
Status WorkerGroup::Start(const WorkerExecutable& executable, std::span<WorkerCgroup* const> cgroups,
    std::span<const WorkerBootstrap> bootstraps, const WorkerEnvironment& environment, NcclBootstrapBroker& broker) {
  if (state_ != WorkerGroupState::kEmpty) return Status::FailedPrecondition("Worker group is single-use");
  if ((bootstraps.size() != 2 && bootstraps.size() != 4 && bootstraps.size() != 8) ||
      cgroups.size() != bootstraps.size()) return Status::InvalidArgument("Worker group topology invalid");
  try {
    const auto admitted_executable = executable.Descriptor();
    if (!admitted_executable.ok()) return admitted_executable.status();
    const auto baseline = bootstraps[0].Encode(); if (!baseline.ok()) return baseline.status();
    const auto broker_id = broker.BorrowId(); if (!broker_id.ok()) return broker_id.status();
    if (*broker_id != bootstraps[0].nccl_id)
      return Status::FailedPrecondition("Rank bootstrap ID differs from live broker");
    std::array<struct stat, 8> groups{};
    std::array<int, 8> group_fds{};
    for (std::size_t i = 0; i < bootstraps.size(); ++i) {
      const auto& b = bootstraps[i];
      auto valid = b.Validate(); if (!valid.ok()) return valid;
      if (!cgroups[i]) return Status::InvalidArgument("Worker cgroup owner is absent");
      if (cgroups[i] == broker.cgroup_owner()) return Status::InvalidArgument("Rank cannot share broker cgroup");
      auto group = cgroups[i]->ReadyDescriptor(); if (!group.ok()) return group.status();
      auto empty = cgroups[i]->Empty(); if (!empty.ok()) return empty.status();
      if (!*empty) return Status::FailedPrecondition("Worker cgroup is populated before launch");
      group_fds[i] = *group;
      if (b.rank != i || b.world != bootstraps.size() || b.supervisor_pid != static_cast<std::uint32_t>(::getpid()) ||
          b.supervisor_uid != static_cast<std::uint32_t>(::geteuid()) ||
          ::fstat(group_fds[i], &groups[i]) || !S_ISDIR(groups[i].st_mode))
        return Status::InvalidArgument("Worker group rank, parent or cgroup invalid");
      // Only rank-local placement, payload budgets and shard identity may vary.
      // Every other field (including sampling, deadlines, NCCL ID and backend
      // lock) must match exactly, via canonical encoding rather than memcmp.
      auto common = b;
      common.rank = bootstraps[0].rank; common.device = bootstraps[0].device;
      common.device_budget = bootstraps[0].device_budget; common.host_budget = bootstraps[0].host_budget;
      common.artifact_directory = bootstraps[0].artifact_directory;
      common.weight_manifest_sha256 = bootstraps[0].weight_manifest_sha256;
      common.endpoints = bootstraps[0].endpoints;
      auto encoded = common.Encode(); if (!encoded.ok()) return encoded.status();
      if (*encoded != *baseline) return Status::FailedPrecondition("Worker group sequence configuration differs across ranks");
      for (std::size_t j = 0; j < i; ++j) {
        if (b.device == bootstraps[j].device || (groups[i].st_dev == groups[j].st_dev && groups[i].st_ino == groups[j].st_ino))
          return Status::InvalidArgument("Worker group requires distinct devices and cgroups");
        for (const auto& name : b.endpoints) for (const auto& prior : bootstraps[j].endpoints)
          if (name == prior) return Status::InvalidArgument("Worker group endpoint names collide");
      }
    }
    deadline_ = Clock::time_point(std::chrono::duration_cast<Clock::duration>(
        std::chrono::nanoseconds(static_cast<std::int64_t>(bootstraps[0].startup_ns))));
    if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Worker group startup expired");
    if (broker.lifetime_deadline() <= deadline_)
      return Status::InvalidArgument("NCCL broker lifetime does not cover rank startup");
    sequence_deadline_ = Clock::time_point(std::chrono::duration_cast<Clock::duration>(
        std::chrono::nanoseconds(static_cast<std::int64_t>(bootstraps[0].sequence_ns))));
    identity_ = bootstraps[0].identity; sampling_ = bootstraps[0].sampling;
    prompt_tokens_ = bootstraps[0].prompt_tokens; retirement_ms_ = bootstraps[0].retirement_ms;
    maximum_positions_ = bootstraps[0].maximum_positions;
    world_ = static_cast<std::uint32_t>(bootstraps.size());
    for (unsigned i = 0; i < world_; ++i) cgroups_[i] = cgroups[i];
    state_ = WorkerGroupState::kFailed;
    broker_ = &broker;
    std::array<std::optional<SealedWorkerBootstrap>, 8> sealed;
    for (unsigned i = 0; i < world_; ++i) {
      const auto& b = bootstraps[i];
      const std::array<std::string_view, 4> names{b.endpoints[0], b.endpoints[1], b.endpoints[2], b.endpoints[3]};
      auto channel = RankChannels::Listen(names, i, world_); if (!channel.ok()) return channel.status();
      channels_[i] = std::move(*channel);
      auto data = SealedWorkerBootstrap::Create(b); if (!data.ok()) return data.status();
      sealed[i].emplace(std::move(*data));
    }
    const std::array<std::string, 1> arguments{"pih-v41-rank-worker"};
    for (unsigned i = 0; i < world_; ++i) {
      auto started = workers_[i].Start(executable, sealed[i]->descriptor(), group_fds[i], arguments, environment, deadline_);
      if (!started.ok()) return started;
      auto accepted = channels_[i]->BeginAccept(workers_[i].binding().pid, bootstraps[i].supervisor_uid, deadline_);
      if (!accepted.ok()) return accepted;
      sealed[i].reset();
    }
    state_ = WorkerGroupState::kConnecting;
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    state_ = WorkerGroupState::kFailed; return Status::ResourceExhausted("Worker group startup allocation failed; inspect launch records");
  }
}
Result<bool> WorkerGroup::Poll() {
  try {
  if (state_ == WorkerGroupState::kReady) return true;
  if (state_ != WorkerGroupState::kConnecting) return Status::FailedPrecondition("Worker group is not connecting");
  const auto fail = [&](Status status) -> Result<bool> { state_ = WorkerGroupState::kFailed; return status; };
  const auto broker_ready = broker_->Poll();
  if (!broker_ready.ok()) return fail(broker_ready.status());
  if (!*broker_ready) return fail(Status::FailedPrecondition("NCCL broker lost admitted ID"));
  if (Clock::now() >= deadline_) return fail(Status::DeadlineExceeded("Worker group channel admission expired"));
  for (unsigned i = 0; i < world_; ++i) {
    if (connected_ & (1U << i)) continue;
    auto exec = workers_[i].PollExec(); if (!exec.ok()) return fail(exec.status());
    if (!*exec) continue;
    auto channel = channels_[i]->Poll(); if (!channel.ok()) return fail(channel.status());
    if (!*channel) continue;
    auto views = channels_[i]->Views(); if (!views.ok()) return fail(views.status());
    requests_[i] = &views->requests; receipts_[i] = &views->receipts;
    commands_[i] = &views->commands; notices_[i] = &views->notices;
    connected_ |= 1U << i;
  }
  if (connected_ != (1U << world_) - 1U) return false;
  std::array<RankProcessBinding, 8> ranks{};
  for (unsigned i = 0; i < world_; ++i) ranks[i] = workers_[i].binding();
  auto watch = RankProcessWatch::Attach({ranks.data(), world_}); if (!watch.ok()) return fail(watch.status());
  processes_.emplace(std::move(*watch));
  if (Clock::now() >= deadline_) return fail(Status::DeadlineExceeded("Worker group admission completed late"));
  state_ = WorkerGroupState::kReady;
  return true;
  } catch (const std::bad_alloc&) {
    state_ = WorkerGroupState::kFailed;
    return Status::ResourceExhausted("Worker group admission allocation failed; retain launch records");
  }
}
Result<std::unique_ptr<GenerationSession>> WorkerGroup::StartGeneration(TokenLedger& ledger,
    std::span<const std::uint32_t> prompt, std::span<const std::string_view> bytes,
    std::uint64_t first_plan, std::chrono::milliseconds grace) {
  if (state_ != WorkerGroupState::kReady) return Status::FailedPrecondition("Worker group cannot transfer process custody");
  const auto& id = ledger.identity(); const auto& p = ledger.initial_sampling();
  if (ledger.failed() || !ledger.records().empty() || ledger.pending_output() ||
        ledger.prompt_tokens() != prompt_tokens_ || prompt.size() != prompt_tokens_ ||
        ledger.maximum_completion_tokens() > maximum_positions_ - prompt_tokens_ ||
      id.epoch != identity_.epoch || id.plan_seq || id.sequence_generation != identity_.sequence_generation ||
      id.sampling_config_id != identity_.sampling_config_id ||
      std::bit_cast<std::uint32_t>(p.temperature) != std::bit_cast<std::uint32_t>(sampling_.temperature) ||
      std::bit_cast<std::uint32_t>(p.top_p) != std::bit_cast<std::uint32_t>(sampling_.top_p) ||
      p.top_k != sampling_.top_k || p.seed != sampling_.seed || p.ordinal || p.suppressed_count ||
      p.logprobs != sampling_.logprobs || p.top_count != sampling_.top_count ||
      !std::equal(std::begin(p.suppressed), std::end(p.suppressed), std::begin(sampling_.suppressed)) ||
      grace.count() < 0 || grace >= std::chrono::milliseconds(retirement_ms_))
    return Status::InvalidArgument("Generation ledger differs from launched worker sequence or retirement policy");
  const auto live = processes_->CheckLive();
  if (!live.ok()) { state_ = WorkerGroupState::kFailed; return live; }
  if (Clock::now() >= deadline_) {
    state_ = WorkerGroupState::kFailed; return Status::DeadlineExceeded("Worker group custody transfer expired");
  }
  state_ = WorkerGroupState::kFailed;
  try {
    auto loop = GenerationLoop::Create(ledger, prompt, bytes, {requests_.data(), world_},
        {receipts_.data(), world_}, *processes_, first_plan, sequence_deadline_);
    if (!loop.ok()) return loop.status();
      auto session = GenerationSession::Create(std::move(*loop), {commands_.data(), world_},
          {notices_.data(), world_}, {cgroups_.data(), world_}, *broker_, deadline_, grace,
        std::chrono::milliseconds(retirement_ms_));
    if (!session.ok()) return session.status();
    state_ = WorkerGroupState::kHandedOff;
    return std::move(*session);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Generation construction failed; worker group retains startup retirement");
  }
}
std::uint32_t WorkerGroup::spawned_mask() const noexcept {
  std::uint32_t mask = 0;
  for (unsigned i = 0; i < world_; ++i) if (workers_[i].binding().pid > 0) mask |= 1U << i;
  return mask;
}
std::uint32_t WorkerGroup::reaped_mask() const noexcept {
  std::uint32_t mask = processes_ ? processes_->reaped_mask() : 0;
  for (unsigned i = 0; i < world_; ++i) if (workers_[i].state() == WorkerSpawnState::kReaped) mask |= 1U << i;
  return mask;
}
Status WorkerGroup::BeginStartupRetirement(Clock::time_point deadline) {
  if (state_ == WorkerGroupState::kHandedOff || state_ == WorkerGroupState::kEmpty ||
      state_ == WorkerGroupState::kRetired || state_ == WorkerGroupState::kRetiring || deadline <= Clock::now())
    return Status::FailedPrecondition("Worker group startup retirement not permitted");
  deadline_ = deadline; state_ = WorkerGroupState::kRetiring;
  return Status::Ok();
}
Result<bool> WorkerGroup::PollStartupRetirement() {
  if (state_ == WorkerGroupState::kRetired) return true;
  if (state_ != WorkerGroupState::kRetiring) return Status::FailedPrecondition("Worker group has no startup retirement");
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Partial worker group retirement expired");
  bool complete = true; Status first = Status::Ok();
  bool broker_complete = true; Status broker_failure = Status::Ok();
  if (broker_ && broker_->state() != NcclBrokerState::kRetired) {
    if (broker_->state() != NcclBrokerState::kRetiring) {
      const auto started = broker_->BeginRetirement(deadline_);
      if (!started.ok()) broker_failure = started;
    }
    if (broker_->state() == NcclBrokerState::kRetiring) {
      const auto retired = broker_->PollRetirement();
      if (!retired.ok()) { if (broker_failure.ok()) broker_failure = retired.status(); }
    }
    if (broker_->state() != NcclBrokerState::kRetired) broker_complete = false;
  }
  // Descendants are contained by cgroup.kill, not merely the leader pidfd.
  for (unsigned i = 0; i < world_; ++i) {
    if (cgroups_[i]->removed()) continue;
    const auto killed = cgroups_[i]->Kill();
    if (!killed.ok()) { if (first.ok()) first = killed; complete = false; }
  }
  for (unsigned i = 0; i < world_; ++i) {
    if (workers_[i].binding().pid <= 0 || workers_[i].state() == WorkerSpawnState::kReaped) continue;
    const auto killed = workers_[i].Kill();
    if (!killed.ok()) { if (first.ok()) first = killed; complete = false; continue; }
    const auto reaped = workers_[i].PollKilled(deadline_);
    if (!reaped.ok()) { if (first.ok()) first = reaped.status(); complete = false; }
    else if (!*reaped) complete = false;
  }
  if (!first.ok()) return first;
  if (complete) {
    for (unsigned i = 0; i < world_; ++i) {
      if (cgroups_[i]->removed()) continue;
      auto empty = cgroups_[i]->Empty();
      if (!empty.ok()) { if (first.ok()) first = empty.status(); complete = false; continue; }
      if (!*empty) { complete = false; continue; }
      const auto removed = cgroups_[i]->Remove();
      if (!removed.ok()) { if (first.ok()) first = removed; complete = false; }
    }
  }
  if (!first.ok()) return first;
  if (!broker_failure.ok()) return broker_failure;
  complete = complete && broker_complete;
  if (complete) state_ = WorkerGroupState::kRetired;
  return complete;
}
}  // namespace pih::deepseek_v41
