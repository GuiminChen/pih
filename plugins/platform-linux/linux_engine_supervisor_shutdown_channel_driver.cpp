#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_engine_supervisor_shutdown_channel_driver.h"

#include <cerrno>
#include <limits>
#include <string>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

}  // namespace

Result<LinuxEngineSupervisorShutdownChannelDriver>
LinuxEngineSupervisorShutdownChannelDriver::Create(
    std::int32_t inherited_fd, std::uint64_t expected_supervisor_pid,
    std::uint32_t expected_supervisor_uid) {
  if (inherited_fd < 0 || expected_supervisor_pid == 0 ||
      expected_supervisor_pid >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "engine supervisor shutdown channel identity is invalid");
  }
  const int owned_fd = ::fcntl(inherited_fd, F_DUPFD_CLOEXEC, 0);
  if (owned_fd < 0) return failure("duplicate supervisor shutdown channel");
  const auto reject = [owned_fd](Status status)
      -> Result<LinuxEngineSupervisorShutdownChannelDriver> {
    (void)::close(owned_fd);
    return status;
  };
  int socket_type = 0;
  socklen_t socket_type_bytes = sizeof(socket_type);
  if (::getsockopt(owned_fd, SOL_SOCKET, SO_TYPE, &socket_type,
                   &socket_type_bytes) != 0) {
    return reject(failure("getsockopt supervisor shutdown SO_TYPE"));
  }
  if (socket_type_bytes != sizeof(socket_type) ||
      socket_type != SOCK_SEQPACKET) {
    return reject(Status::FailedPrecondition(
        "engine supervisor shutdown channel is not SOCK_SEQPACKET"));
  }
  struct ucred credentials {};
  socklen_t credential_bytes = sizeof(credentials);
  if (::getsockopt(owned_fd, SOL_SOCKET, SO_PEERCRED, &credentials,
                   &credential_bytes) != 0) {
    return reject(failure("getsockopt supervisor shutdown SO_PEERCRED"));
  }
  if (credential_bytes != sizeof(credentials) || credentials.pid <= 0 ||
      static_cast<std::uint64_t>(credentials.pid) != expected_supervisor_pid ||
      static_cast<std::uint32_t>(credentials.uid) != expected_supervisor_uid) {
    return reject(Status::FailedPrecondition(
        "engine supervisor shutdown channel peer identity drifted"));
  }
  return LinuxEngineSupervisorShutdownChannelDriver(owned_fd);
}

LinuxEngineSupervisorShutdownChannelDriver::
    LinuxEngineSupervisorShutdownChannelDriver(
        LinuxEngineSupervisorShutdownChannelDriver&& other) noexcept
    : fd_(other.fd_) {
  other.fd_ = -1;
}

LinuxEngineSupervisorShutdownChannelDriver&
LinuxEngineSupervisorShutdownChannelDriver::operator=(
    LinuxEngineSupervisorShutdownChannelDriver&& other) noexcept {
  if (this == &other) return *this;
  close_owned();
  fd_ = other.fd_;
  other.fd_ = -1;
  return *this;
}

LinuxEngineSupervisorShutdownChannelDriver::
    ~LinuxEngineSupervisorShutdownChannelDriver() {
  close_owned();
}

void LinuxEngineSupervisorShutdownChannelDriver::close_owned() noexcept {
  if (fd_ >= 0) (void)::close(fd_);
  fd_ = -1;
}

Status LinuxEngineSupervisorShutdownChannelDriver::send_request(
    std::span<const std::byte> frame) {
  if (frame.size() != kEngineSupervisorShutdownFrameBytes) {
    return Status::InvalidArgument(
        "engine supervisor shutdown request frame size is invalid");
  }
  const auto written = ::send(fd_, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "engine supervisor shutdown request channel is backpressured");
  }
  if (written < 0) return failure("send supervisor shutdown request");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "engine supervisor shutdown request was partially sent");
  }
  return Status::Ok();
}

Result<std::optional<std::vector<std::byte>>>
LinuxEngineSupervisorShutdownChannelDriver::poll_ack() {
  std::vector<std::byte> frame(kEngineSupervisorShutdownFrameBytes);
  const auto received =
      ::recv(fd_, frame.data(), frame.size(), MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return std::optional<std::vector<std::byte>>{};
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown channel closed before acknowledgement");
  }
  if (received < 0) return failure("receive supervisor shutdown acknowledgement");
  if (received != static_cast<ssize_t>(frame.size())) {
    return Status::InvalidArgument(
        "engine supervisor shutdown acknowledgement frame size is invalid");
  }
  return std::optional<std::vector<std::byte>>{std::move(frame)};
}

Result<std::unique_ptr<LinuxEngineSupervisorShutdownClient>>
LinuxEngineSupervisorShutdownClient::Create(
    std::int32_t inherited_fd, std::uint64_t expected_supervisor_pid,
    std::uint32_t expected_supervisor_uid, std::uint64_t engine_generation,
    std::uint64_t engine_epoch, std::uint64_t deadline_ns) {
  if (engine_generation == 0 || engine_epoch == 0 || deadline_ns == 0) {
    return Status::InvalidArgument(
        "engine supervisor shutdown transaction bounds are invalid");
  }
  auto driver_value = LinuxEngineSupervisorShutdownChannelDriver::Create(
      inherited_fd, expected_supervisor_pid, expected_supervisor_uid);
  if (!driver_value.ok()) return driver_value.status();
  auto driver = std::make_unique<LinuxEngineSupervisorShutdownChannelDriver>(
      std::move(*driver_value));
  EngineSupervisorShutdownRequest request{
      engine_generation, engine_epoch, 1, deadline_ns,
      EngineSupervisorShutdownRequestKind::kForceStop};
  auto channel_value = EngineSupervisorShutdownChannel::Create(request, *driver);
  if (!channel_value.ok()) return channel_value.status();
  auto channel = std::make_unique<EngineSupervisorShutdownChannel>(
      std::move(*channel_value));
  return std::unique_ptr<LinuxEngineSupervisorShutdownClient>(
      new LinuxEngineSupervisorShutdownClient(engine_generation, engine_epoch,
                                              std::move(driver),
                                              std::move(channel)));
}

}  // namespace pih
