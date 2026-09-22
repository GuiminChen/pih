#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_engine_supervisor_shutdown_server_driver.h"

#include <cerrno>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pih {
namespace {

Status server_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

}  // namespace

Result<LinuxEngineSupervisorShutdownServerDriver>
LinuxEngineSupervisorShutdownServerDriver::Create(
    std::int32_t inherited_fd, std::uint64_t expected_engine_pid,
    std::uint32_t expected_engine_uid) {
  if (inherited_fd < 0 || expected_engine_pid == 0 ||
      expected_engine_pid >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "engine supervisor shutdown server identity is invalid");
  }
  const int owned_fd = ::fcntl(inherited_fd, F_DUPFD_CLOEXEC, 0);
  if (owned_fd < 0) {
    return server_failure("duplicate supervisor shutdown server channel");
  }
  const auto reject = [owned_fd](Status status)
      -> Result<LinuxEngineSupervisorShutdownServerDriver> {
    (void)::close(owned_fd);
    return status;
  };

  int socket_type = 0;
  socklen_t socket_type_bytes = sizeof(socket_type);
  if (::getsockopt(owned_fd, SOL_SOCKET, SO_TYPE, &socket_type,
                   &socket_type_bytes) != 0) {
    return reject(server_failure(
        "getsockopt supervisor shutdown server SO_TYPE"));
  }
  if (socket_type_bytes != sizeof(socket_type) ||
      socket_type != SOCK_SEQPACKET) {
    return reject(Status::FailedPrecondition(
        "engine supervisor shutdown server channel is not SOCK_SEQPACKET"));
  }

  const int pass_credentials = 1;
  if (::setsockopt(owned_fd, SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   sizeof(pass_credentials)) != 0) {
    return reject(server_failure(
        "setsockopt supervisor shutdown server SO_PASSCRED"));
  }
  return LinuxEngineSupervisorShutdownServerDriver(
      owned_fd, expected_engine_pid, expected_engine_uid);
}

LinuxEngineSupervisorShutdownServerDriver::
    LinuxEngineSupervisorShutdownServerDriver(
    LinuxEngineSupervisorShutdownServerDriver&& other) noexcept
    : fd_(other.fd_), expected_engine_pid_(other.expected_engine_pid_),
      expected_engine_uid_(other.expected_engine_uid_) {
  other.fd_ = -1;
  other.expected_engine_pid_ = 0;
  other.expected_engine_uid_ = 0;
}

LinuxEngineSupervisorShutdownServerDriver&
LinuxEngineSupervisorShutdownServerDriver::operator=(
    LinuxEngineSupervisorShutdownServerDriver&& other) noexcept {
  if (this == &other) return *this;
  close_owned();
  fd_ = other.fd_;
  expected_engine_pid_ = other.expected_engine_pid_;
  expected_engine_uid_ = other.expected_engine_uid_;
  other.fd_ = -1;
  other.expected_engine_pid_ = 0;
  other.expected_engine_uid_ = 0;
  return *this;
}

LinuxEngineSupervisorShutdownServerDriver::
    ~LinuxEngineSupervisorShutdownServerDriver() {
  close_owned();
}

void LinuxEngineSupervisorShutdownServerDriver::close_owned() noexcept {
  if (fd_ >= 0) (void)::close(fd_);
  fd_ = -1;
}

Result<std::optional<std::vector<std::byte>>>
LinuxEngineSupervisorShutdownServerDriver::poll_request() {
  std::vector<std::byte> frame(kEngineSupervisorShutdownFrameBytes);
  struct iovec payload {
    frame.data(), frame.size()
  };
  alignas(struct cmsghdr)
      std::array<std::byte, CMSG_SPACE(sizeof(struct ucred))> control{};
  struct msghdr message {};
  message.msg_iov = &payload;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  const auto received =
      ::recvmsg(fd_, &message, MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return std::optional<std::vector<std::byte>>{};
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown server channel closed before request");
  }
  if (received < 0) {
    return server_failure("receive supervisor shutdown request");
  }
  if (received != static_cast<ssize_t>(frame.size()) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    return Status::InvalidArgument(
        "engine supervisor shutdown request frame size is invalid");
  }
  const struct ucred* credentials = nullptr;
  for (auto* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET ||
        header->cmsg_type != SCM_CREDENTIALS) {
      continue;
    }
    if (credentials != nullptr ||
        header->cmsg_len != CMSG_LEN(sizeof(struct ucred))) {
      return Status::FailedPrecondition(
          "engine supervisor shutdown request credentials are ambiguous");
    }
    credentials = reinterpret_cast<const struct ucred*>(CMSG_DATA(header));
  }
  if (credentials == nullptr || credentials->pid <= 0 ||
      static_cast<std::uint64_t>(credentials->pid) != expected_engine_pid_ ||
      static_cast<std::uint32_t>(credentials->uid) != expected_engine_uid_) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown request sender identity drifted");
  }
  return std::optional<std::vector<std::byte>>{std::move(frame)};
}

Status LinuxEngineSupervisorShutdownServerDriver::send_ack(
    std::span<const std::byte> frame) {
  if (frame.size() != kEngineSupervisorShutdownFrameBytes) {
    return Status::InvalidArgument(
        "engine supervisor shutdown acknowledgement frame size is invalid");
  }
  const auto written = ::send(fd_, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "engine supervisor shutdown acknowledgement channel is backpressured");
  }
  if (written < 0) {
    return server_failure("send supervisor shutdown acknowledgement");
  }
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "engine supervisor shutdown acknowledgement was partially sent");
  }
  return Status::Ok();
}

Result<std::unique_ptr<LinuxEngineSupervisorShutdownService>>
LinuxEngineSupervisorShutdownService::Create(
    std::int32_t inherited_fd, std::uint64_t expected_engine_pid,
    std::uint32_t expected_engine_uid, std::uint64_t engine_generation,
    std::uint64_t engine_epoch) {
  if (engine_generation == 0 || engine_epoch == 0) {
    return Status::InvalidArgument(
        "engine supervisor shutdown service identity is invalid");
  }
  auto driver_value = LinuxEngineSupervisorShutdownServerDriver::Create(
      inherited_fd, expected_engine_pid, expected_engine_uid);
  if (!driver_value.ok()) return driver_value.status();
  auto driver =
      std::make_unique<LinuxEngineSupervisorShutdownServerDriver>(
          std::move(*driver_value));
  auto server_value = EngineSupervisorShutdownServer::Create(*driver);
  if (!server_value.ok()) return server_value.status();
  auto server = std::make_unique<EngineSupervisorShutdownServer>(
      std::move(*server_value));
  auto handler_value = EngineSupervisorShutdownHandler::Create(
      engine_generation, engine_epoch);
  if (!handler_value.ok()) return handler_value.status();
  auto handler = std::make_unique<EngineSupervisorShutdownHandler>(
      std::move(*handler_value));
  return std::unique_ptr<LinuxEngineSupervisorShutdownService>(
      new LinuxEngineSupervisorShutdownService(
          engine_generation, engine_epoch, std::move(driver),
          std::move(server), std::move(handler)));
}

}  // namespace pih
