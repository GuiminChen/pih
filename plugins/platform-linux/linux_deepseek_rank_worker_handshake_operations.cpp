#include "pih/platform/linux/linux_deepseek_rank_worker_handshake_operations.h"

#include "pih/core/checked_math.h"

#include <cerrno>
#include <fstream>
#include <limits>
#include <string>

#include <sys/prctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

namespace pih {
namespace {

Status failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

Result<std::uint64_t> pidfd_target(int fd) {
  std::ifstream input("/proc/self/fdinfo/" + std::to_string(fd));
  std::string label;
  std::uint64_t value = 0;
  while (input >> label >> value) {
    if (label == "Pid:") return value;
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return Status::FailedPrecondition(
      "DeepSeek worker controller pidfd has no process identity");
}

}  // namespace

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankWorkerHandshakeOperations::receive_challenge(
    std::int32_t control_fd) {
  std::vector<std::byte> frame(kDeepSeekRankChallengeBytes);
  const auto received = ::recv(control_fd, frame.data(), frame.size(),
                               MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return std::optional<std::vector<std::byte>>{};
  if (received == 0)
    return Status::FailedPrecondition(
        "DeepSeek worker control channel closed before challenge");
  if (received < 0) return failure("recv DeepSeek rank challenge");
  if (received != static_cast<ssize_t>(frame.size()))
    return Status::InvalidArgument("DeepSeek rank challenge frame size is invalid");
  return std::optional<std::vector<std::byte>>{std::move(frame)};
}

Result<DeepSeekRankExecObservation>
LinuxDeepSeekRankWorkerHandshakeOperations::collect_observation(
    const DeepSeekRankWorkerArguments& arguments) {
  int death_signal = 0;
  if (::prctl(PR_GET_PDEATHSIG, &death_signal) != 0)
    return failure("PR_GET_PDEATHSIG");
  auto controller = pidfd_target(arguments.controller_pidfd);
  if (!controller.ok()) return controller.status();
  int socket_type = 0;
  socklen_t socket_type_bytes = sizeof(socket_type);
  if (::getsockopt(arguments.control_fd, SOL_SOCKET, SO_TYPE, &socket_type,
                   &socket_type_bytes) != 0)
    return failure("getsockopt DeepSeek rank control socket");
  auto device = device_probe_->current_physical_device_uuid_commitment();
  if (!device.ok()) return device.status();
  auto ordinal = device_probe_->startup_device_ordinal();
  if (!ordinal.ok()) return ordinal.status();
  DeepSeekRankExecObservation observation;
  observation.argument_manifest = arguments.manifest;
  observation.actual_process_identity =
      static_cast<std::uint64_t>(::getpid());
  observation.actual_parent_process_identity =
      static_cast<std::uint64_t>(::getppid());
  observation.controller_pidfd_target_identity = *controller;
  observation.actual_physical_device_identity =
      arguments.manifest.physical_device_identity;
  observation.actual_physical_device_uuid_commitment = *device;
  observation.actual_startup_device_ordinal = *ordinal;
  observation.parent_death_signal = death_signal;
  observation.control_channel_is_seqpacket = socket_type == SOCK_SEQPACKET;
  observation.challenge_received_on_control_channel = true;
  return observation;
}

Status LinuxDeepSeekRankWorkerHandshakeOperations::send_ready(
    std::int32_t control_fd, std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankReadyBytes)
    return Status::InvalidArgument("DeepSeek rank ready frame size is invalid");
  const auto written = ::send(control_fd, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return Status::Unavailable("DeepSeek rank ready channel is backpressured");
  if (written < 0) return failure("send DeepSeek rank ready");
  if (written != static_cast<ssize_t>(frame.size()))
    return Status::Internal("DeepSeek rank ready frame was partially sent");
  return Status::Ok();
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankWorkerHandshakeOperations::receive_authority(
    std::int32_t control_fd) {
  std::vector<std::byte> frame(
      kDeepSeekRankPostExecResourceAuthorityBytes);
  const auto received = ::recv(control_fd, frame.data(), frame.size(),
                               MSG_DONTWAIT | MSG_TRUNC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return std::optional<std::vector<std::byte>>{};
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "DeepSeek worker control channel closed before resource authority");
  }
  if (received < 0) return failure("recv DeepSeek rank resource authority");
  if (received != static_cast<ssize_t>(frame.size())) {
    return Status::InvalidArgument(
        "DeepSeek rank resource authority frame size is invalid");
  }
  return std::optional<std::vector<std::byte>>{std::move(frame)};
}

Result<std::uint64_t>
LinuxDeepSeekRankWorkerHandshakeOperations::monotonic_now_ns() {
  timespec now{};
  if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return failure("clock_gettime CLOCK_MONOTONIC");
  }
  if (now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1'000'000'000L) {
    return Status::Internal("Linux monotonic clock value is invalid");
  }
  auto seconds = checked_mul_u64(
      static_cast<std::uint64_t>(now.tv_sec), 1'000'000'000ULL);
  if (!seconds.ok()) return seconds.status();
  return checked_add_u64(*seconds,
                         static_cast<std::uint64_t>(now.tv_nsec));
}

Status LinuxDeepSeekRankWorkerHandshakeOperations::send_observation(
    std::int32_t control_fd, std::span<const std::byte> frame) {
  if (frame.size() != kDeepSeekRankPostExecResourceObservationBytes) {
    return Status::InvalidArgument(
        "DeepSeek rank resource observation frame size is invalid");
  }
  const auto written = ::send(control_fd, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek rank resource observation channel is backpressured");
  }
  if (written < 0) return failure("send DeepSeek rank resource observation");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek rank resource observation frame was partially sent");
  }
  return Status::Ok();
}

}  // namespace pih
