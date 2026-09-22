#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_deepseek_rank_post_mapping_resource_operations.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

namespace pih {
namespace {

Status system_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

bool same_handle(const DeepSeekRankProcessHandle& left,
                 const DeepSeekRankProcessHandle& right) noexcept {
  return left.process_identity == right.process_identity &&
         left.pidfd_identity == right.pidfd_identity &&
         left.control_identity == right.control_identity;
}

Status validate_controller_socket(int descriptor) {
  int socket_type = 0;
  socklen_t type_size = sizeof(socket_type);
  int pass_credentials = 0;
  socklen_t pass_size = sizeof(pass_credentials);
  if (descriptor < 0 ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &socket_type,
                   &type_size) != 0 ||
      type_size != sizeof(socket_type) || socket_type != SOCK_SEQPACKET ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   &pass_size) != 0 ||
      pass_size != sizeof(pass_credentials) || pass_credentials != 1) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping controller socket is not credentialed seqpacket");
  }
  return Status::Ok();
}

Status validate_worker_socket(int descriptor,
                              std::uint64_t expected_controller) {
  if (expected_controller == 0 ||
      expected_controller >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping controller process identity is invalid");
  }
  auto status = validate_controller_socket(descriptor);
  if (!status.ok()) return status;
  struct ucred peer {};
  socklen_t peer_size = sizeof(peer);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &peer,
                   &peer_size) != 0 ||
      peer_size != sizeof(peer) ||
      peer.pid != static_cast<pid_t>(expected_controller) ||
      peer.uid != ::geteuid() || peer.gid != ::getegid()) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping worker socket peer differs");
  }
  return Status::Ok();
}

struct CredentialObservation final {
  std::uint32_t count = 0;
  struct ucred credential {};
};

Status collect_credential_without_rights(
    msghdr& message, CredentialObservation* result) {
  if (result == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping credential output is absent");
  }
  bool invalid = false;
  for (auto* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level == SOL_SOCKET &&
        header->cmsg_type == SCM_RIGHTS &&
        header->cmsg_len >= CMSG_LEN(sizeof(int))) {
      const auto payload_bytes = header->cmsg_len - CMSG_LEN(0);
      const auto count = payload_bytes / sizeof(int);
      const auto* descriptors =
          reinterpret_cast<const int*>(CMSG_DATA(header));
      for (std::size_t index = 0; index < count; ++index) {
        if (descriptors[index] >= 0) (void)::close(descriptors[index]);
      }
      invalid = true;
      continue;
    }
    if (header->cmsg_level != SOL_SOCKET ||
        header->cmsg_type != SCM_CREDENTIALS ||
        header->cmsg_len != CMSG_LEN(sizeof(struct ucred))) {
      invalid = true;
      continue;
    }
    ++result->count;
    std::memcpy(&result->credential, CMSG_DATA(header),
                sizeof(struct ucred));
  }
  if (invalid) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping ancillary record is invalid");
  }
  return Status::Ok();
}

Result<std::vector<std::byte>> receive_credentialed_packet(
    int socket, std::size_t expected_bytes,
    std::uint64_t expected_process_identity) {
  if (expected_process_identity == 0 ||
      expected_process_identity >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping packet process identity is invalid");
  }
  std::vector<std::byte> frame(expected_bytes);
  iovec vector{frame.data(), frame.size()};
  alignas(cmsghdr) std::array<
      std::byte,
      CMSG_SPACE(sizeof(struct ucred)) + CMSG_SPACE(sizeof(int) * 16U)>
      control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  const auto received = ::recvmsg(
      socket, &message,
      MSG_DONTWAIT | MSG_TRUNC | MSG_CMSG_CLOEXEC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek post-mapping packet is pending");
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping control socket closed");
  }
  if (received < 0) {
    return system_failure("recvmsg post-mapping resource packet");
  }
  CredentialObservation credential;
  auto status = collect_credential_without_rights(message, &credential);
  if (!status.ok()) return status;
  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      received != static_cast<ssize_t>(expected_bytes) ||
      credential.count != 1 ||
      credential.credential.pid !=
          static_cast<pid_t>(expected_process_identity) ||
      credential.credential.uid != ::geteuid() ||
      credential.credential.gid != ::getegid()) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping packet identity is invalid");
  }
  return frame;
}

Result<std::uint64_t> monotonic_now() {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0 ||
      value.tv_sec < 0 || value.tv_nsec < 0 ||
      value.tv_nsec >= 1'000'000'000L) {
    return system_failure("clock_gettime post-mapping resource");
  }
  return static_cast<std::uint64_t>(value.tv_sec) *
             UINT64_C(1'000'000'000) +
         static_cast<std::uint64_t>(value.tv_nsec);
}

}  // namespace

Result<LinuxDeepSeekRankPostMappingResourceControllerOperations>
LinuxDeepSeekRankPostMappingResourceControllerOperations::Create(
    LinuxDeepSeekRankProcessDriver& driver,
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessHandle> ordered_handles) {
  if (!supervisor.ready() || supervisor.failed() ||
      ordered_handles.empty() || ordered_handles.size() > 4) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping controller lacks a ready process set");
  }
  if (ordered_handles.size() < 4 &&
      (supervisor.process_handle(
           static_cast<std::uint32_t>(ordered_handles.size())) != nullptr ||
       supervisor.exec_ready(
           static_cast<std::uint32_t>(ordered_handles.size())) != nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping controller process set is truncated");
  }
  for (std::uint32_t rank = 0; rank < ordered_handles.size(); ++rank) {
    const auto* supervised = supervisor.process_handle(rank);
    const auto* ready = supervisor.exec_ready(rank);
    if (supervised == nullptr || ready == nullptr ||
        !same_handle(*supervised, ordered_handles[rank]) ||
        ready->receipt.rank != rank ||
        ready->receipt.process_identity !=
            ordered_handles[rank].process_identity) {
      return Status::FailedPrecondition(
          "DeepSeek post-mapping controller process identity differs");
    }
    auto* process = driver.find(ordered_handles[rank]);
    if (process == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek post-mapping driver handle differs");
    }
    auto status = validate_controller_socket(process->control);
    if (!status.ok()) return status;
  }
  return LinuxDeepSeekRankPostMappingResourceControllerOperations(
      driver,
      std::vector<DeepSeekRankProcessHandle>(ordered_handles.begin(),
                                             ordered_handles.end()));
}

Result<std::uint32_t>
LinuxDeepSeekRankPostMappingResourceControllerOperations::rank_for(
    const DeepSeekRankProcessHandle& handle) const {
  for (std::uint32_t rank = 0; rank < handles_.size(); ++rank) {
    if (same_handle(handles_[rank], handle)) return rank;
  }
  return Status::FailedPrecondition(
      "DeepSeek post-mapping process handle is unknown");
}

Result<std::int32_t>
LinuxDeepSeekRankPostMappingResourceControllerOperations::control_fd(
    std::uint32_t rank) const {
  if (rank >= handles_.size()) {
    return Status::InvalidArgument(
        "DeepSeek post-mapping controller rank is invalid");
  }
  auto* process = driver_->find(handles_[rank]);
  if (process == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping controller handle disappeared");
  }
  auto status = validate_controller_socket(process->control);
  if (!status.ok()) return status;
  return process->control;
}

Status LinuxDeepSeekRankPostMappingResourceControllerOperations::
    send_authority(const DeepSeekRankProcessHandle& handle,
                   std::span<const std::byte> frame) {
  auto rank = rank_for(handle);
  if (!rank.ok()) return rank.status();
  auto authority = decode_deepseek_rank_post_mapping_resource_authority(frame);
  if (!authority.ok()) return authority.status();
  if (authority->rank != *rank ||
      authority->process_identity != handle.process_identity ||
      authority->pidfd_identity != handle.pidfd_identity ||
      authority->control_identity != handle.control_identity) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping authority handle differs");
  }
  auto socket = control_fd(*rank);
  if (!socket.ok()) return socket.status();
  const auto written = ::send(
      *socket, frame.data(), frame.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek post-mapping authority socket is backpressured");
  }
  if (written < 0) {
    return system_failure("send post-mapping resource authority");
  }
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek post-mapping authority was partially sent");
  }
  return Status::Ok();
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankPostMappingResourceControllerOperations::poll_observation(
    const DeepSeekRankProcessHandle& handle) {
  auto rank = rank_for(handle);
  if (!rank.ok()) return rank.status();
  auto socket = control_fd(*rank);
  if (!socket.ok()) return socket.status();
  auto frame = receive_credentialed_packet(
      *socket, kDeepSeekRankPostMappingResourceObservationFrameBytes,
      handle.process_identity);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return frame.status();
  }
  auto observation =
      decode_deepseek_rank_post_mapping_resource_observation(*frame);
  if (!observation.ok()) return observation.status();
  if (observation->resources.rank != *rank ||
      observation->resources.process_identity != handle.process_identity) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping observation handle differs");
  }
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Result<std::uint64_t>
LinuxDeepSeekRankPostMappingResourceControllerOperations::
    monotonic_now_ns() {
  return monotonic_now();
}

Result<LinuxDeepSeekRankPostMappingResourceReporterOperations>
LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
    std::int32_t control_fd,
    std::uint64_t expected_controller_process_identity) {
  auto status = validate_worker_socket(
      control_fd, expected_controller_process_identity);
  if (!status.ok()) return status;
  return LinuxDeepSeekRankPostMappingResourceReporterOperations(
      control_fd, expected_controller_process_identity);
}

Status LinuxDeepSeekRankPostMappingResourceReporterOperations::
    validate_bound_control(std::int32_t control_fd) const {
  if (control_fd != control_fd_) {
    return Status::FailedPrecondition(
        "DeepSeek post-mapping worker control descriptor changed");
  }
  return validate_worker_socket(
      control_fd_, controller_process_identity_);
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankPostMappingResourceReporterOperations::receive_authority(
    std::int32_t control_fd) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  auto frame = receive_credentialed_packet(
      control_fd_, kDeepSeekRankPostMappingResourceAuthorityFrameBytes,
      controller_process_identity_);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return frame.status();
  }
  auto authority =
      decode_deepseek_rank_post_mapping_resource_authority(*frame);
  if (!authority.ok()) return authority.status();
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Result<std::uint64_t>
LinuxDeepSeekRankPostMappingResourceReporterOperations::monotonic_now_ns() {
  return monotonic_now();
}

Status LinuxDeepSeekRankPostMappingResourceReporterOperations::
    send_observation(std::int32_t control_fd,
                     std::span<const std::byte> frame) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  auto observation =
      decode_deepseek_rank_post_mapping_resource_observation(frame);
  if (!observation.ok()) return observation.status();
  const auto written = ::send(
      control_fd_, frame.data(), frame.size(),
      MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek post-mapping observation socket is backpressured");
  }
  if (written < 0) {
    return system_failure("send post-mapping resource observation");
  }
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek post-mapping observation was partially sent");
  }
  return Status::Ok();
}

}  // namespace pih
