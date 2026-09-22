#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "rank_socket.h"
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
Result<sockaddr_un> Address(std::string_view name) {
  // 32 characters leaves space for a 128-bit random hex nonce. Entropy and
  // uniqueness are a supervisor obligation, not inferred from string length.
  if (name.size() < 32 || name.size() > 96)
    return Status::InvalidArgument("Rank socket abstract name length invalid");
  for (const auto c : name)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
      return Status::InvalidArgument("Rank socket abstract name contains invalid character");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path + 1, name.data(), name.size());
  return address;
}
socklen_t Length(std::string_view name) {
  return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());
}
}
RankSocket::RankSocket(RankSocket&& other) noexcept
    : fd_(other.fd_), state_(other.state_), pid_(other.pid_), uid_(other.uid_), deadline_(other.deadline_) {
  other.fd_ = -1; other.state_ = State::kFailed;
}
RankSocket::~RankSocket() { if (fd_ >= 0) ::close(fd_); }
Status RankSocket::Authenticate(std::int32_t pid, std::uint32_t uid) const {
  ucred peer{}; socklen_t bytes = sizeof(peer);
  if (::getsockopt(fd_, SOL_SOCKET, SO_PEERCRED, &peer, &bytes) || bytes != sizeof(peer) ||
      peer.pid != pid || peer.uid != uid)
    return Status::FailedPrecondition("Rank socket authenticated peer differs from spawn identity");
  sockaddr_un address{}; bytes = sizeof(address);
  if (::getpeername(fd_, reinterpret_cast<sockaddr*>(&address), &bytes) || address.sun_family != AF_UNIX)
    return Status::FailedPrecondition("Rank socket has no connected Unix peer");
  return Status::Ok();
}
Result<RankSocket> RankSocket::Listen(std::string_view name) {
  auto address = Address(name); if (!address.ok()) return address.status();
  RankSocket socket;
  socket.fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (socket.fd_ < 0) return Status::Unavailable("Cannot create rank listener");
  if (::bind(socket.fd_, reinterpret_cast<const sockaddr*>(&*address), Length(name)) || ::listen(socket.fd_, 1))
    return Status::Unavailable("Cannot bind/listen rank endpoint; name is never replaced");
  socket.state_ = State::kListening;
  return socket;
}
Result<RankSocket> RankSocket::Connect(std::string_view name, std::int32_t pid, std::uint32_t uid, Clock::time_point deadline) {
  auto address = Address(name); if (!address.ok()) return address.status();
  if (pid <= 0 || Clock::now() >= deadline) return Status::InvalidArgument("Rank connection identity or deadline invalid");
  RankSocket socket;
  socket.fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (socket.fd_ < 0) return Status::Unavailable("Cannot create worker connection");
  socket.pid_ = pid; socket.uid_ = uid; socket.deadline_ = deadline;
  if (::connect(socket.fd_, reinterpret_cast<const sockaddr*>(&*address), Length(name)) != 0) {
    // AF_UNIX EAGAIN (backlog full) is not an in-progress connection. Do not
    // return a socket that SO_ERROR=0 could falsely admit as connected.
    if (errno != EINPROGRESS) return Status::Unavailable("Worker connection failed; no automatic reconnect");
    socket.state_ = State::kConnecting;
  } else {
    const auto peer = socket.Authenticate(pid, uid); if (!peer.ok()) return peer;
    if (Clock::now() >= deadline) return Status::DeadlineExceeded("Worker connected after deadline");
    socket.state_ = State::kConnected;
  }
  return socket;
}
Result<bool> RankSocket::PollConnected() {
  if (state_ == State::kConnected) return true;
  if (state_ != State::kConnecting) return Status::FailedPrecondition("Rank socket has no pending connection");
  if (Clock::now() >= deadline_) { state_ = State::kFailed; return Status::DeadlineExceeded("Rank connection expired"); }
  pollfd descriptor{fd_, POLLOUT, 0};
  const int ready = ::poll(&descriptor, 1, 0);
  if (ready == 0 || (ready < 0 && errno == EINTR)) return false;
  state_ = State::kFailed;
  if (ready < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) || !(descriptor.revents & POLLOUT))
    return Status::Unavailable("Rank connection poll failed");
  int error = 0; socklen_t bytes = sizeof(error);
  if (::getsockopt(fd_, SOL_SOCKET, SO_ERROR, &error, &bytes) || bytes != sizeof(error) || error)
    return Status::Unavailable("Rank connection completed with socket error");
  const auto peer = Authenticate(pid_, uid_); if (!peer.ok()) return peer;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Rank connection completed late");
  state_ = State::kConnected;
  return true;
}
Result<std::optional<RankSocket>> RankSocket::PollAccept(std::int32_t pid, std::uint32_t uid, Clock::time_point deadline) {
  if (state_ != State::kListening) return Status::FailedPrecondition("Rank socket is not listening");
  if (pid <= 0) return Status::InvalidArgument("Expected worker PID invalid");
  if (Clock::now() >= deadline) { state_ = State::kFailed; return Status::DeadlineExceeded("Rank accept expired"); }
  RankSocket peer;
  peer.fd_ = ::accept4(fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
  if (peer.fd_ < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return std::optional<RankSocket>{};
    state_ = State::kFailed; return Status::Unavailable("Rank accept failed");
  }
  // Consume the single-use admission even if the first peer is unauthorized.
  // A supervisor may fail the epoch; it must not silently attach another rank.
  state_ = State::kFailed;
  const auto identity = peer.Authenticate(pid, uid); if (!identity.ok()) return identity;
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Rank accepted after deadline");
  peer.state_ = State::kConnected;
  return std::optional<RankSocket>{std::move(peer)};
}
Result<int> RankSocket::ConnectedDescriptor() const {
  if (state_ != State::kConnected || fd_ < 0) return Status::FailedPrecondition("Rank socket is not authenticated connected");
  return fd_;
}
}  // namespace pih::deepseek_v41
