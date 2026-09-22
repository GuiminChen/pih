#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "rank_channel.h"
#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
RankReceiptChannel::RankReceiptChannel(RankReceiptChannel&& other) noexcept
    : fd_(other.fd_), peer_pid_(other.peer_pid_), rank_(other.rank_), world_(other.world_), direction_(other.direction_),
      frame_(other.frame_), offset_(other.offset_), queued_(other.queued_), failed_(other.failed_) {
  other.fd_ = -1; other.failed_ = true;
}
RankReceiptChannel::~RankReceiptChannel() { if (fd_ >= 0) ::close(fd_); }
Result<RankReceiptChannel> RankReceiptChannel::Attach(int fd, std::int32_t pid, std::uint32_t uid,
    std::uint32_t rank, std::uint32_t world, RankChannelDirection direction) {
  if (fd < 0 || pid <= 0 || (world != 2 && world != 4 && world != 8) || rank >= world ||
      (direction != RankChannelDirection::kSend && direction != RankChannelDirection::kReceive))
    return Status::InvalidArgument("Rank channel descriptor or supervisor identity invalid");
  RankReceiptChannel channel;
  channel.fd_ = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (channel.fd_ < 0) return Status::Unavailable("Cannot duplicate rank channel descriptor");
  const int flags = ::fcntl(channel.fd_, F_GETFL);
  int domain = 0, type = 0; socklen_t length = sizeof(int);
  if (flags < 0 || !(flags & O_NONBLOCK) ||
      ::getsockopt(channel.fd_, SOL_SOCKET, SO_DOMAIN, &domain, &length) || length != sizeof(int) || domain != AF_UNIX)
    return Status::InvalidArgument("Rank channel must be a nonblocking Unix socket");
  length = sizeof(int);
  if (::getsockopt(channel.fd_, SOL_SOCKET, SO_TYPE, &type, &length) || length != sizeof(int) || type != SOCK_STREAM)
    return Status::InvalidArgument("Rank channel must be a stream socket");
  sockaddr_storage peer{}; length = sizeof(peer);
  if (::getpeername(channel.fd_, reinterpret_cast<sockaddr*>(&peer), &length) || peer.ss_family != AF_UNIX)
    return Status::FailedPrecondition("Rank channel has no connected Unix peer");
  ucred credentials{}; length = sizeof(credentials);
  if (::getsockopt(channel.fd_, SOL_SOCKET, SO_PEERCRED, &credentials, &length) || length != sizeof(credentials) ||
      credentials.pid != pid || credentials.uid != uid)
    return Status::FailedPrecondition("Rank channel peer differs from supervisor process identity");
  channel.peer_pid_ = pid; channel.rank_ = rank; channel.world_ = world; channel.direction_ = direction; return channel;
}
Status RankReceiptChannel::Queue(const RankStepReceipt& receipt) {
  if (failed_ || fd_ < 0 || direction_ != RankChannelDirection::kSend || queued_)
    return Status::FailedPrecondition("Rank channel cannot queue another receipt");
  if (receipt.rank() != rank_ || receipt.world() != world_)
    return Status::InvalidArgument("Receipt differs from channel rank");
  frame_ = receipt.Encode(); offset_ = 0; queued_ = true; return Status::Ok();
}
Result<bool> RankReceiptChannel::PollSend(Clock::time_point deadline) {
  if (failed_ || fd_ < 0 || direction_ != RankChannelDirection::kSend || !queued_)
    return Status::FailedPrecondition("Rank channel has no pending send");
  if (Clock::now() >= deadline) { failed_ = true; return Status::DeadlineExceeded("Rank receipt send timed out"); }
  const auto sent = ::send(fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT | MSG_NOSIGNAL);
  if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (sent <= 0) { failed_ = true; return Status::Unavailable("Rank receipt channel send failed"); }
  offset_ += static_cast<std::size_t>(sent);
  if (offset_ != frame_.size()) return false;
  if (Clock::now() >= deadline) { failed_ = true; return Status::DeadlineExceeded("Rank receipt send completed after deadline"); }
  queued_ = false; offset_ = 0; return true;
}
Result<bool> RankReceiptChannel::PollReceive(RankCommitGate& gate, Clock::time_point deadline) {
  if (failed_ || fd_ < 0 || direction_ != RankChannelDirection::kReceive) {
    gate.Fail(); return Status::FailedPrecondition("Rank receipt channel cannot receive");
  }
  if (Clock::now() >= deadline) { failed_ = true; gate.Fail(); return Status::DeadlineExceeded("Rank receipt receive timed out"); }
  const auto received = ::recv(fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (received <= 0) { failed_ = true; gate.Fail(); return Status::Unavailable("Rank receipt channel disconnected or failed"); }
  offset_ += static_cast<std::size_t>(received);
  if (offset_ != frame_.size()) return false;
  if (Clock::now() >= deadline) { failed_ = true; gate.Fail(); return Status::DeadlineExceeded("Rank receipt received after deadline"); }
  const auto accepted = gate.SubmitWire(rank_, frame_);
  if (!accepted.ok()) { failed_ = true; return accepted; }
  offset_ = 0; return true;
}
Status RankReceiptChannel::CheckQuiet(RankCommitGate& gate) {
  if (failed_ || fd_ < 0 || direction_ != RankChannelDirection::kReceive || offset_) {
    gate.Fail(); return Status::FailedPrecondition("Rank receipt boundary is not idle");
  }
  std::uint8_t byte = 0;
  const auto result = ::recv(fd_, &byte, 1, MSG_DONTWAIT | MSG_PEEK);
  if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return Status::Ok();
  // Even EINTR cannot certify the pre-commit boundary as quiet.
  failed_ = true; gate.Fail(); return Status::FailedPrecondition("Rank receipt boundary has data, disconnect or an unverified read");
}
Result<bool> RankReceiptChannel::PollTerminal(const SamplingIdentity& identity, std::uint64_t ordinal,
    std::uint32_t processed, Clock::time_point deadline) {
  if (failed_ || fd_ < 0 || direction_ != RankChannelDirection::kReceive)
    return Status::FailedPrecondition("Terminal receipt channel cannot receive");
  if (Clock::now() >= deadline) { failed_ = true; return Status::DeadlineExceeded("Terminal receipt timed out"); }
  const auto received = ::recv(fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (received <= 0) { failed_ = true; return Status::Unavailable("Terminal receipt disconnected or failed"); }
  offset_ += static_cast<std::size_t>(received); if (offset_ != frame_.size()) return false;
  RankStepReceipt expected; expected.identity_ = identity; expected.ordinal_ = ordinal; expected.processed_ = processed;
  expected.rank_ = rank_; expected.world_ = world_; expected.terminal_ = true;
  if (frame_ != expected.Encode() || Clock::now() >= deadline) {
    failed_ = true; return Status::FailedPrecondition("Terminal receipt identity, payload or deadline mismatch");
  }
  offset_ = 0; return true;
}
}  // namespace pih::deepseek_v41
