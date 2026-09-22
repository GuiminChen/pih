#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "rank_process_watch.h"
#include <cerrno>
#include <csignal>
#include <new>
#include <charconv>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/wait.h>

namespace pih::deepseek_v41 {
namespace {
Status Identity(int fd, std::int32_t expected) {
  const auto path = "/proc/self/fdinfo/" + std::to_string(fd);
  const int info = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (info < 0) return Status::Unavailable("Cannot read rank pidfd identity");
  std::array<char, 8192> bytes{}; std::size_t used = 0;
  while (used < bytes.size()) {
    const auto n = ::read(info, bytes.data() + used, bytes.size() - used);
    if (n < 0) { ::close(info); return Status::Unavailable("Rank pidfd identity read failed"); }
    if (!n) break;
    used += static_cast<std::size_t>(n);
  }
  ::close(info);
  if (used == bytes.size()) return Status::InvalidArgument("Rank pidfd identity exceeds bound");
  std::string_view remaining(bytes.data(), used);
  while (!remaining.empty()) {
    const auto end = remaining.find('\n');
    auto line = remaining.substr(0, end);
    if (line.starts_with("Pid:")) {
      line.remove_prefix(4);
      while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.remove_prefix(1);
      std::int32_t pid = 0;
      const auto parsed = std::from_chars(line.data(), line.data() + line.size(), pid);
      if (parsed.ec == std::errc{} && parsed.ptr == line.data() + line.size() && pid == expected)
        return Status::Ok();
      return Status::FailedPrecondition("Rank pidfd process identity mismatch or process already reaped");
    }
    if (end == std::string_view::npos) break;
    remaining.remove_prefix(end + 1);
  }
  return Status::FailedPrecondition("Descriptor is not an identified rank pidfd");
}
}
RankProcessWatch::RankProcessWatch(RankProcessWatch&& other) noexcept
    : ranks_(other.ranks_), world_(other.world_), reaped_(other.reaped_),
      reaping_(other.reaping_), normal_exit_failed_(other.normal_exit_failed_), failed_(other.failed_) {
  other.world_ = 0; other.failed_ = true;
}
RankProcessWatch::~RankProcessWatch() { for (unsigned i = 0; i < world_; ++i) if (ranks_[i].pidfd >= 0) ::close(ranks_[i].pidfd); }
Result<RankProcessWatch> RankProcessWatch::Attach(std::span<const RankProcessBinding> ranks) {
  if (ranks.size() != 2 && ranks.size() != 4 && ranks.size() != 8)
    return Status::InvalidArgument("Rank pidfd world invalid");
  RankProcessWatch watch;
  for (std::size_t i = 0; i < ranks.size(); ++i) {
    if (ranks[i].pidfd < 0 || ranks[i].pid <= 0) return Status::InvalidArgument("Rank pidfd binding invalid");
    for (std::size_t j = 0; j < i; ++j) if (ranks[j].pid == ranks[i].pid)
      return Status::InvalidArgument("Rank processes must be distinct");
    const int owned = ::fcntl(ranks[i].pidfd, F_DUPFD_CLOEXEC, 0);
    if (owned < 0) return Status::Unavailable("Cannot retain rank pidfd");
    watch.ranks_[watch.world_++] = {owned, ranks[i].pid};
    const auto identity = Identity(owned, ranks[i].pid); if (!identity.ok()) return identity;
  }
  const auto live = watch.CheckLive(); if (!live.ok()) return live;
  return watch;
}
Status RankProcessWatch::CheckLive() {
  if (failed_ || reaping_ || !world_) return Status::FailedPrecondition("Rank process watch failed, reaping or moved from");
  std::array<pollfd, 8> descriptors{};
  for (unsigned i = 0; i < world_; ++i) descriptors[i] = {ranks_[i].pidfd, POLLIN, 0};
  const int ready = ::poll(descriptors.data(), world_, 0);
  if (ready == 0) return Status::Ok();
  failed_ = true;
  return ready < 0 ? Status::Unavailable("Cannot certify rank process liveness") :
      Status::FailedPrecondition("A rank exited or its pidfd became invalid");
}
Result<bool> RankProcessWatch::PollNormalExit(std::chrono::steady_clock::time_point deadline) {
  if (!world_ || normal_exit_failed_ || (failed_ && !reaping_))
    return Status::FailedPrecondition("Normal rank reaping is unavailable");
  reaping_ = true; failed_ = true; // Permanently close inference admission.
  const auto fail = [&](Status status) -> Result<bool> { normal_exit_failed_ = true; return status; };
  if (std::chrono::steady_clock::now() >= deadline)
    return fail(Status::DeadlineExceeded("Normal rank exit deadline expired"));
  for (unsigned i = 0; i < world_; ++i) {
    const auto bit = 1U << i; if (reaped_ & bit) continue;
    siginfo_t info{};
    if (::waitid(P_PIDFD, static_cast<id_t>(ranks_[i].pidfd), &info, WEXITED | WNOHANG) != 0) {
      if (errno == EINTR) continue;
      return fail(Status::FailedPrecondition("Normal pidfd reap requires exclusive parent/subreaper ownership"));
    }
    if (!info.si_pid) continue;
    if (info.si_pid != ranks_[i].pid ||
        (info.si_code != CLD_EXITED && info.si_code != CLD_KILLED && info.si_code != CLD_DUMPED))
      return fail(Status::FailedPrecondition("Normal rank reap identity or classification mismatch"));
    // Record even an abnormal exit before reporting failure. Fault retirement
    // must not waitid/reap this already-consumed child again.
    reaped_ |= bit;
    if (info.si_code != CLD_EXITED || info.si_status != 0)
      return fail(Status::FailedPrecondition("Released worker exited abnormally or with nonzero status"));
  }
  if (std::chrono::steady_clock::now() >= deadline)
    return fail(Status::DeadlineExceeded("Normal rank exit completed after deadline"));
  return reaped_ == (1U << world_) - 1U;
}
Result<std::unique_ptr<RankProcessRetirement>> RankProcessRetirement::Create(RankProcessWatch& processes, TokenLedger& ledger,
    Clock::time_point kill_after, Clock::time_point deadline) {
  if (!processes.world_ || deadline <= Clock::now() || kill_after >= deadline)
    return Status::InvalidArgument("Rank retirement world or deadlines invalid");
  try {
    auto retirement = std::unique_ptr<RankProcessRetirement>(new RankProcessRetirement);
    retirement->kill_after_ = kill_after; retirement->deadline_ = deadline;
    retirement->reaped_ = processes.reaped_;
    for (unsigned i = 0; i < processes.world_; ++i) {
      const int fd = ::fcntl(processes.ranks_[i].pidfd, F_DUPFD_CLOEXEC, 0);
      if (fd < 0) return Status::Unavailable("Cannot retain rank retirement pidfd");
      retirement->ranks_[retirement->world_++] = {fd, processes.ranks_[i].pid};
    }
    processes.failed_ = true; // This process set can no longer admit commits.
    retirement->ledger_ = &ledger;
    if (const auto output = ledger.pending_output()) retirement->pending_plan_ = output->credit.plan_sequence;
    ledger.Fail();
    return retirement;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Rank retirement allocation failed"); }
}
RankProcessRetirement::~RankProcessRetirement() {
  for (unsigned i = 0; i < world_; ++i) if (ranks_[i].pidfd >= 0) ::close(ranks_[i].pidfd);
}
Result<RankRetirementState> RankProcessRetirement::Poll() {
  if (state_ == RankRetirementState::kFailed) return Status::FailedPrecondition("Rank retirement failed; retain reconciliation records");
  if (state_ == RankRetirementState::kComplete) return state_;
  const auto fail = [&](Status status) -> Result<RankRetirementState> { state_ = RankRetirementState::kFailed; return status; };
  if (Clock::now() >= deadline_) return fail(Status::DeadlineExceeded("Rank retirement deadline expired"));
  for (unsigned i = 0; i < world_; ++i) {
    const auto bit = 1U << i; if (reaped_ & bit) continue;
    siginfo_t info{};
    if (::waitid(P_PIDFD, static_cast<id_t>(ranks_[i].pidfd), &info, WEXITED | WNOHANG) != 0) {
      if (errno == EINTR) continue;
      return fail(Status::FailedPrecondition("Rank pidfd reap failed; exclusive parent/subreaper ownership required"));
    }
    if (info.si_pid) {
      if (info.si_pid != ranks_[i].pid || (info.si_code != CLD_EXITED && info.si_code != CLD_KILLED && info.si_code != CLD_DUMPED))
        return fail(Status::FailedPrecondition("Rank reap identity or exit classification mismatch"));
      reaped_ |= bit; continue;
    }
    if (Clock::now() >= deadline_) return fail(Status::DeadlineExceeded("Rank retirement deadline expired before signal"));
    const bool kill = Clock::now() >= kill_after_;
    auto& signalled = kill ? killed_ : terminated_;
    if (!(signalled & bit)) {
      if (::syscall(SYS_pidfd_send_signal, ranks_[i].pidfd, kill ? SIGKILL : SIGTERM, nullptr, 0) != 0 && errno != ESRCH) {
        if (errno == EINTR) continue;
        return fail(Status::Unavailable("Rank pidfd signal failed; no numeric PID fallback"));
      }
      signalled |= bit;
    }
  }
  if (reaped_ == ((1U << world_) - 1)) state_ = RankRetirementState::kComplete;
  else if (Clock::now() >= kill_after_) state_ = RankRetirementState::kKilling;
  return state_;
}
Status RankProcessRetirement::DiscardOutput() {
  if (state_ != RankRetirementState::kComplete)
    return Status::FailedPrecondition("Output cannot be discarded until every rank is reaped");
  if (!pending_plan_) return Status::Ok();
  const auto discarded = ledger_->AbortRetired(pending_plan_); if (!discarded.ok()) return discarded;
  pending_plan_ = 0; return Status::Ok();
}
}  // namespace pih::deepseek_v41
