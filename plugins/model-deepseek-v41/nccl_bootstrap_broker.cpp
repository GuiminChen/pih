#include "nccl_bootstrap_broker.h"
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
void NcclBootstrapBroker::Clear() noexcept {
  volatile std::byte* bytes = id_.data();
  for (std::size_t i = 0; i < id_.size(); ++i) bytes[i] = std::byte{};
}
NcclBootstrapBroker::~NcclBootstrapBroker() { Clear(); if (socket_ >= 0) ::close(socket_); }
Status NcclBootstrapBroker::Start(const WorkerExecutable& helper, WorkerCgroup& group,
    const WorkerEnvironment& environment, Clock::time_point startup, Clock::time_point lifetime) {
  if (state_ != NcclBrokerState::kEmpty || startup <= Clock::now() || lifetime <= startup)
    return Status::InvalidArgument("NCCL broker is single-use with ordered deadlines");
  auto directory = group.ReadyDescriptor(); if (!directory.ok()) return directory.status();
  auto empty = group.Empty(); if (!empty.ok()) return empty.status();
  if (!*empty) return Status::FailedPrecondition("NCCL broker requires an empty dedicated cgroup");
  cgroup_ = &group; startup_ = startup; lifetime_ = lifetime;
  state_ = NcclBrokerState::kFailed;
  int pair[2]{-1, -1};
  if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0, pair))
    return Status::Unavailable("Cannot allocate private NCCL broker channel");
  socket_ = pair[0];
  const auto started = process_.StartNcclBroker(helper, pair[1], *directory, environment, startup, lifetime);
  ::close(pair[1]);
  if (!started.ok()) return started;
  state_ = NcclBrokerState::kStarting;
  return Status::Ok();
}
Status NcclBootstrapBroker::Live() const {
  const auto binding = process_.binding();
  if (binding.pidfd < 0 || Clock::now() >= lifetime_)
    return Status::DeadlineExceeded("NCCL broker is absent or expired");
  pollfd fd{binding.pidfd, POLLIN, 0};
  if (::poll(&fd, 1, 0) != 0) return Status::FailedPrecondition("NCCL broker process no longer live");
  return Status::Ok();
}
Result<bool> NcclBootstrapBroker::Poll() {
  if (state_ != NcclBrokerState::kStarting && state_ != NcclBrokerState::kReady)
    return Status::FailedPrecondition("NCCL broker is not awaiting or holding an ID");
  const auto fail = [&](Status error) -> Result<bool> { state_ = NcclBrokerState::kFailed; Clear(); return error; };
  const auto live = Live(); if (!live.ok()) return fail(live);
  if (state_ == NcclBrokerState::kStarting) {
    if (Clock::now() >= startup_) return fail(Status::DeadlineExceeded("NCCL broker ID admission expired"));
    auto exec = process_.PollExec(); if (!exec.ok()) return fail(exec.status());
    if (!*exec) return false;
  }
  std::array<std::byte, 144> packet{};
  const auto received = ::recv(socket_, packet.data(), packet.size(), MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EINTR)) return state_ == NcclBrokerState::kReady;
  if (state_ == NcclBrokerState::kReady || received != static_cast<ssize_t>(packet.size()))
    return fail(Status::FailedPrecondition("NCCL broker packet missing, repeated or malformed"));
  auto id = NcclBootstrapChannel::Decode(packet); if (!id.ok()) return fail(id.status());
  if (Clock::now() >= startup_) return fail(Status::DeadlineExceeded("NCCL broker ID arrived late"));
  id_ = *id; state_ = NcclBrokerState::kReady;
  return true;
}
Result<NcclBootstrapChannel::Id> NcclBootstrapBroker::BorrowId() {
  auto ready = Poll(); if (!ready.ok()) return ready.status();
  if (!*ready) return Status::FailedPrecondition("NCCL bootstrap ID not ready");
  return id_;
}
Status NcclBootstrapBroker::BeginRetirement(Clock::time_point deadline) {
  if (!cgroup_ || state_ == NcclBrokerState::kEmpty || state_ == NcclBrokerState::kRetiring ||
      state_ == NcclBrokerState::kRetired || deadline <= Clock::now())
    return Status::FailedPrecondition("NCCL broker cannot enter retirement");
  Clear(); retirement_ = deadline; state_ = NcclBrokerState::kRetiring;
  return Status::Ok();
}
Result<bool> NcclBootstrapBroker::PollRetirement() {
  if (state_ == NcclBrokerState::kRetired) return true;
  if (state_ != NcclBrokerState::kRetiring) return Status::FailedPrecondition("NCCL broker retirement not started");
  if (Clock::now() >= retirement_) return Status::DeadlineExceeded("NCCL broker retirement expired");
  const auto contained = cgroup_->Kill();
  Status killed = Status::Ok();
  if (process_.binding().pidfd >= 0 && process_.state() != WorkerSpawnState::kReaped) killed = process_.Kill();
  if (!killed.ok()) return killed;
  if (!contained.ok()) return contained;
  if (process_.binding().pidfd >= 0) {
    auto reaped = process_.PollKilled(retirement_); if (!reaped.ok()) return reaped.status();
    if (!*reaped) return false;
  }
  auto empty = cgroup_->Empty(); if (!empty.ok()) return empty.status();
  if (!*empty) return false;
  const auto removed = cgroup_->Remove(); if (!removed.ok()) return removed;
  if (socket_ >= 0) { ::close(socket_); socket_ = -1; }
  state_ = NcclBrokerState::kRetired;
  return true;
}
}  // namespace pih::deepseek_v41
