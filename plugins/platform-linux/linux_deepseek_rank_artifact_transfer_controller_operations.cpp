#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_deepseek_rank_artifact_transfer_controller_operations.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "pih/model/deepseek_rank_artifact_descriptor_batch.h"

namespace pih {
namespace {

Status system_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

Status validate_control_socket(int descriptor) {
  int socket_type = 0;
  socklen_t size = sizeof(socket_type);
  int pass_credentials = 0;
  socklen_t pass_size = sizeof(pass_credentials);
  if (descriptor < 0 ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &socket_type, &size) != 0 ||
      size != sizeof(socket_type) || socket_type != SOCK_SEQPACKET ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   &pass_size) != 0 ||
      pass_size != sizeof(pass_credentials) || pass_credentials != 1) {
    return Status::FailedPrecondition(
        "DeepSeek artifact transfer control socket is not credentialed seqpacket");
  }
  return Status::Ok();
}

bool same_handle(const DeepSeekRankProcessHandle& left,
                 const DeepSeekRankProcessHandle& right) noexcept {
  return left.process_identity == right.process_identity &&
         left.pidfd_identity == right.pidfd_identity &&
         left.control_identity == right.control_identity;
}

struct CredentialObservation final {
  std::uint32_t count = 0;
  struct ucred credential {};
};

Status collect_credentials(msghdr& message,
                           CredentialObservation* observation) {
  if (observation == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek artifact ACK credential output is absent");
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
    ++observation->count;
    std::memcpy(&observation->credential, CMSG_DATA(header),
                sizeof(struct ucred));
  }
  if (invalid) {
    return Status::FailedPrecondition(
        "DeepSeek artifact ACK ancillary record is invalid");
  }
  return Status::Ok();
}

}  // namespace

LinuxDeepSeekRankArtifactTransferControllerOperations::
    LinuxDeepSeekRankArtifactTransferControllerOperations(
        LinuxDeepSeekRankProcessDriver& driver,
        DeepSeekRankProcessSupervisor& supervisor,
        std::vector<DeepSeekRankProcessHandle> handles,
        std::uint64_t engine_epoch,
        std::uint64_t worker_generation) noexcept
    : driver_(&driver), supervisor_(&supervisor),
      handles_(std::move(handles)), engine_epoch_(engine_epoch),
      worker_generation_(worker_generation) {}

Result<LinuxDeepSeekRankArtifactTransferControllerOperations>
LinuxDeepSeekRankArtifactTransferControllerOperations::Create(
    LinuxDeepSeekRankProcessDriver& driver,
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessHandle> ordered_handles) {
  if (!supervisor.ready() || supervisor.failed() ||
      ordered_handles.empty() || ordered_handles.size() > 4) {
    return Status::FailedPrecondition(
        "DeepSeek artifact controller operations lack a ready supervisor");
  }
  if (ordered_handles.size() < 4 &&
      (supervisor.process_handle(
           static_cast<std::uint32_t>(ordered_handles.size())) != nullptr ||
       supervisor.exec_ready(
           static_cast<std::uint32_t>(ordered_handles.size())) != nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek artifact controller process set is truncated");
  }
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  for (std::uint32_t rank = 0; rank < ordered_handles.size(); ++rank) {
    const auto* supervised = supervisor.process_handle(rank);
    const auto* ready = supervisor.exec_ready(rank);
    if (supervised == nullptr || ready == nullptr ||
        ordered_handles[rank].process_identity == 0 ||
        ordered_handles[rank].process_identity >
            static_cast<std::uint64_t>(
                std::numeric_limits<pid_t>::max()) ||
        !same_handle(*supervised, ordered_handles[rank]) ||
        ready->receipt.rank != rank ||
        (rank != 0 &&
         (ready->receipt.engine_epoch != engine_epoch ||
          ready->receipt.worker_generation != worker_generation))) {
      return Status::FailedPrecondition(
          "DeepSeek artifact controller process set differs");
    }
    if (rank == 0) {
      engine_epoch = ready->receipt.engine_epoch;
      worker_generation = ready->receipt.worker_generation;
    }
    auto* process = driver.find(ordered_handles[rank]);
    if (process == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek artifact controller driver handle differs");
    }
    const auto socket_status = validate_control_socket(process->control);
    if (!socket_status.ok()) return socket_status;
  }
  return LinuxDeepSeekRankArtifactTransferControllerOperations(
      driver, supervisor,
      std::vector<DeepSeekRankProcessHandle>(ordered_handles.begin(),
                                             ordered_handles.end()),
      engine_epoch, worker_generation);
}

Result<std::int32_t>
LinuxDeepSeekRankArtifactTransferControllerOperations::control_fd(
    std::uint32_t rank) const {
  if (rank >= handles_.size()) {
    return Status::InvalidArgument(
        "DeepSeek artifact controller rank is out of range");
  }
  auto* process = driver_->find(handles_[rank]);
  if (process == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek artifact controller process handle disappeared");
  }
  auto status = validate_control_socket(process->control);
  if (!status.ok()) return status;
  return process->control;
}

Status LinuxDeepSeekRankArtifactTransferControllerOperations::send_manifest(
    std::uint32_t rank, std::span<const std::byte> manifest_frame) {
  if (manifest_frame.size() !=
      kDeepSeekRankArtifactTransferManifestBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact transfer manifest frame size is invalid");
  }
  auto descriptor = control_fd(rank);
  if (!descriptor.ok()) return descriptor.status();
  const auto written = ::send(
      *descriptor, manifest_frame.data(), manifest_frame.size(),
      MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek artifact transfer manifest socket is backpressured");
  }
  if (written < 0) return system_failure("send artifact transfer manifest");
  if (written != static_cast<ssize_t>(manifest_frame.size())) {
    return Status::Internal(
        "DeepSeek artifact transfer manifest was partially sent");
  }
  return Status::Ok();
}

Status
LinuxDeepSeekRankArtifactTransferControllerOperations::send_descriptor_batch(
    const DeepSeekRankArtifactDescriptorBatchView& batch) {
  if (batch.rank >= handles_.size() ||
      batch.descriptors.size() != batch.expectations.size() ||
      batch.descriptors.empty() ||
      batch.descriptors.size() >
          kDeepSeekRankArtifactTransferDescriptorBatchMaximum) {
    return Status::InvalidArgument(
        "DeepSeek artifact descriptor send batch is invalid");
  }
  auto typed_batch = DeepSeekRankArtifactDescriptorBatch::FromView(batch);
  if (!typed_batch.ok()) return typed_batch.status();
  auto payload = encode_deepseek_rank_artifact_descriptor_batch(*typed_batch);
  if (!payload.ok()) return payload.status();
  auto socket = control_fd(batch.rank);
  if (!socket.ok()) return socket.status();

  std::array<int, kDeepSeekRankArtifactTransferDescriptorBatchMaximum>
      descriptors{};
  for (std::size_t index = 0; index < batch.descriptors.size(); ++index) {
    const auto descriptor =
        batch.descriptors[index].descriptor
            .linux_file_descriptor_for_transfer();
    const auto descriptor_flags = ::fcntl(descriptor, F_GETFD);
    const auto status_flags = ::fcntl(descriptor, F_GETFL);
    if (descriptor < 0 || descriptor_flags < 0 ||
        (descriptor_flags & FD_CLOEXEC) == 0 || status_flags < 0 ||
        (status_flags & O_ACCMODE) != O_RDONLY) {
      return Status::FailedPrecondition(
          "DeepSeek artifact sender descriptor flags are invalid");
    }
    descriptors[index] = descriptor;
  }

  iovec vector{payload->data(), payload->size()};
  alignas(cmsghdr) std::array<
      std::byte,
      CMSG_SPACE(sizeof(int) *
                 kDeepSeekRankArtifactTransferDescriptorBatchMaximum)>
      control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = CMSG_SPACE(
      sizeof(int) * batch.descriptors.size());
  auto* header = CMSG_FIRSTHDR(&message);
  if (header == nullptr) {
    return Status::Internal(
        "DeepSeek artifact sender ancillary buffer is invalid");
  }
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(int) * batch.descriptors.size());
  std::memcpy(CMSG_DATA(header), descriptors.data(),
              sizeof(int) * batch.descriptors.size());
  const auto written = ::sendmsg(
      *socket, &message, MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek artifact descriptor socket is backpressured");
  }
  if (written < 0) return system_failure("sendmsg artifact descriptors");
  if (written != static_cast<ssize_t>(payload->size())) {
    return Status::Internal(
        "DeepSeek artifact descriptor packet was partially sent");
  }
  return Status::Ok();
}

Result<std::uint64_t>
LinuxDeepSeekRankArtifactTransferControllerOperations::monotonic_now_ns() {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0 ||
      value.tv_sec < 0 || value.tv_nsec < 0 ||
      value.tv_nsec >= 1'000'000'000L) {
    return system_failure("clock_gettime artifact controller");
  }
  return static_cast<std::uint64_t>(value.tv_sec) *
             UINT64_C(1'000'000'000) +
         static_cast<std::uint64_t>(value.tv_nsec);
}

Result<std::optional<std::array<
    std::byte, kDeepSeekRankArtifactTransferAckBytes>>>
LinuxDeepSeekRankArtifactTransferControllerOperations::poll_ack(
    std::uint32_t rank) {
  auto socket = control_fd(rank);
  if (!socket.ok()) return socket.status();
  std::array<std::byte, kDeepSeekRankArtifactTransferAckBytes> frame{};
  iovec vector{frame.data(), frame.size()};
  alignas(cmsghdr) std::array<
      std::byte,
      CMSG_SPACE(sizeof(struct ucred)) +
          CMSG_SPACE(sizeof(int) *
                     kDeepSeekRankArtifactTransferDescriptorBatchMaximum)>
      control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  const auto received = ::recvmsg(
      *socket, &message, MSG_DONTWAIT | MSG_TRUNC | MSG_CMSG_CLOEXEC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return std::optional<std::array<
        std::byte, kDeepSeekRankArtifactTransferAckBytes>>{};
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "DeepSeek artifact ACK socket closed");
  }
  if (received < 0) return system_failure("recvmsg artifact ACK");
  CredentialObservation credential;
  auto status = collect_credentials(message, &credential);
  if (!status.ok()) return status;
  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      received != static_cast<ssize_t>(frame.size()) ||
      credential.count != 1 ||
      credential.credential.pid !=
          static_cast<pid_t>(handles_[rank].process_identity) ||
      credential.credential.uid != ::geteuid() ||
      credential.credential.gid != ::getegid()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact ACK packet identity is invalid");
  }
  return std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>>{frame};
}

Status
LinuxDeepSeekRankArtifactTransferControllerOperations::abort_generation(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    const Status& cause) {
  if (engine_epoch != engine_epoch_ ||
      worker_generation != worker_generation_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact abort generation identity differs");
  }
  if (supervisor_->failed()) return Status::Ok();
  return supervisor_->abort_post_exec(cause);
}

}  // namespace pih
