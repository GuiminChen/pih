#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_deepseek_rank_artifact_transfer_receiver_operations.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if __has_include(<linux/fsverity.h>)
#include <linux/fsverity.h>
#define PIH_LINUX_ARTIFACT_RECEIVER_HAS_FSVERITY 1
#endif

namespace pih {
namespace {

Status system_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

struct RawDescriptorSet final {
  RawDescriptorSet() = default;
  RawDescriptorSet(const RawDescriptorSet&) = delete;
  RawDescriptorSet& operator=(const RawDescriptorSet&) = delete;
  RawDescriptorSet(RawDescriptorSet&& other) noexcept
      : values(std::move(other.values)) {
    other.values.clear();
  }
  RawDescriptorSet& operator=(RawDescriptorSet&& other) noexcept {
    if (this != &other) {
      close_all();
      values = std::move(other.values);
      other.values.clear();
    }
    return *this;
  }
  ~RawDescriptorSet() { close_all(); }

  void close_all() noexcept {
    for (const auto descriptor : values) {
      if (descriptor >= 0) (void)::close(descriptor);
    }
    values.clear();
  }

  std::vector<int> values;
};

struct ReceivedPacket final {
  std::size_t bytes = 0;
  RawDescriptorSet descriptors;
};

Result<ReceivedPacket> receive_credentialed_packet(
    int socket, std::span<std::byte> payload,
    std::uint64_t expected_process_identity, bool require_descriptors) {
  if (expected_process_identity == 0 ||
      expected_process_identity >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek artifact packet expected process identity is invalid");
  }
  iovec vector{payload.data(), payload.size()};
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
      socket, &message, MSG_DONTWAIT | MSG_TRUNC | MSG_CMSG_CLOEXEC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek artifact receive socket has no packet");
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "DeepSeek artifact receive socket closed");
  }
  if (received < 0) return system_failure("recvmsg artifact packet");

  RawDescriptorSet descriptors;
  std::uint32_t credential_count = 0;
  std::uint32_t rights_count = 0;
  struct ucred credential {};
  bool invalid_ancillary = false;
  for (auto* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET) {
      invalid_ancillary = true;
      continue;
    }
    if (header->cmsg_type == SCM_CREDENTIALS) {
      if (header->cmsg_len != CMSG_LEN(sizeof(struct ucred))) {
        invalid_ancillary = true;
        continue;
      }
      ++credential_count;
      std::memcpy(&credential, CMSG_DATA(header), sizeof(struct ucred));
      continue;
    }
    if (header->cmsg_type == SCM_RIGHTS) {
      if (header->cmsg_len < CMSG_LEN(sizeof(int))) {
        invalid_ancillary = true;
        continue;
      }
      const auto payload_bytes =
          header->cmsg_len - CMSG_LEN(0);
      if (payload_bytes % sizeof(int) != 0) {
        const auto count = payload_bytes / sizeof(int);
        const auto* rejected =
            reinterpret_cast<const int*>(CMSG_DATA(header));
        for (std::size_t index = 0; index < count; ++index) {
          if (rejected[index] >= 0) (void)::close(rejected[index]);
        }
        invalid_ancillary = true;
        continue;
      }
      const auto count = payload_bytes / sizeof(int);
      if (count == 0 ||
          count > kDeepSeekRankArtifactTransferDescriptorBatchMaximum ||
          descriptors.values.size() >
              kDeepSeekRankArtifactTransferDescriptorBatchMaximum - count) {
        const auto* rejected =
            reinterpret_cast<const int*>(CMSG_DATA(header));
        for (std::size_t index = 0; index < count; ++index) {
          if (rejected[index] >= 0) (void)::close(rejected[index]);
        }
        invalid_ancillary = true;
        continue;
      }
      ++rights_count;
      const auto old_size = descriptors.values.size();
      descriptors.values.resize(old_size + count);
      std::memcpy(descriptors.values.data() + old_size, CMSG_DATA(header),
                  count * sizeof(int));
      continue;
    }
    invalid_ancillary = true;
  }

  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      received > static_cast<ssize_t>(payload.size()) ||
      invalid_ancillary || credential_count != 1 ||
      credential.pid != static_cast<pid_t>(expected_process_identity) ||
      credential.uid != ::geteuid() || credential.gid != ::getegid() ||
      (require_descriptors
           ? rights_count != 1 || descriptors.values.empty()
           : rights_count != 0 || !descriptors.values.empty())) {
    return Status::FailedPrecondition(
        "DeepSeek artifact packet credentials or ancillary data are invalid");
  }
  return ReceivedPacket{static_cast<std::size_t>(received),
                        std::move(descriptors)};
}

Status validate_socket(int descriptor,
                       std::uint64_t expected_controller) {
  if (expected_controller == 0 ||
      expected_controller >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek artifact controller identity is invalid");
  }
  int socket_type = 0;
  socklen_t type_size = sizeof(socket_type);
  int pass_credentials = 0;
  socklen_t pass_size = sizeof(pass_credentials);
  struct ucred peer {};
  socklen_t peer_size = sizeof(peer);
  if (descriptor < 0 ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &socket_type,
                   &type_size) != 0 ||
      type_size != sizeof(socket_type) || socket_type != SOCK_SEQPACKET ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   &pass_size) != 0 ||
      pass_size != sizeof(pass_credentials) || pass_credentials != 1 ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &peer,
                   &peer_size) != 0 ||
      peer_size != sizeof(peer) ||
      peer.pid != static_cast<pid_t>(expected_controller) ||
      peer.uid != ::geteuid() || peer.gid != ::getegid()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact worker control socket identity is invalid");
  }
  return Status::Ok();
}

Status validate_received_descriptor(
    int descriptor,
    const DeepSeekRankArtifactDescriptorExpectation& expected) {
  const auto descriptor_flags = ::fcntl(descriptor, F_GETFD);
  const auto status_flags = ::fcntl(descriptor, F_GETFL);
  struct stat observation {};
  if (descriptor < 0 || descriptor_flags < 0 ||
      (descriptor_flags & FD_CLOEXEC) == 0 || status_flags < 0 ||
      (status_flags & O_ACCMODE) != O_RDONLY ||
#ifdef O_PATH
      (status_flags & O_PATH) != 0 ||
#endif
      ::fstat(descriptor, &observation) != 0 ||
      !S_ISREG(observation.st_mode) || observation.st_size < 0) {
    return Status::FailedPrecondition(
        "DeepSeek received artifact descriptor flags are invalid");
  }
  const ArtifactFileIdentity identity{
      static_cast<std::uint64_t>(observation.st_size),
      static_cast<std::uint64_t>(observation.st_dev),
      static_cast<std::uint64_t>(observation.st_ino),
      observation.st_mtim.tv_sec,
      static_cast<std::uint32_t>(observation.st_mtim.tv_nsec)};
  if (identity != expected.identity) {
    return Status::FailedPrecondition(
        "DeepSeek received artifact descriptor fstat differs");
  }
  if (expected.immutability_mode == ArtifactImmutabilityMode::kUncalibrated) {
    if (expected.enforced_digest != Sha256Digest{}) {
      return Status::FailedPrecondition(
          "uncalibrated DeepSeek descriptor has an enforced digest");
    }
    return Status::Ok();
  }
  if (expected.immutability_mode != ArtifactImmutabilityMode::kFsVerity) {
    return Status::FailedPrecondition(
        "DeepSeek received descriptor immutability mode is unsupported");
  }
#ifndef PIH_LINUX_ARTIFACT_RECEIVER_HAS_FSVERITY
  return Status::FailedPrecondition(
      "Linux fs-verity measurement API is unavailable");
#else
  constexpr std::size_t kMaximumDigestBytes = 64;
  alignas(fsverity_digest)
      std::array<unsigned char,
                 sizeof(fsverity_digest) + kMaximumDigestBytes>
          buffer{};
  auto* measurement = reinterpret_cast<fsverity_digest*>(buffer.data());
  measurement->digest_size = kMaximumDigestBytes;
  if (::ioctl(descriptor, FS_IOC_MEASURE_VERITY, measurement) != 0 ||
      measurement->digest_size > kMaximumDigestBytes) {
    return Status::FailedPrecondition(
        "DeepSeek received descriptor lacks fs-verity enforcement");
  }
  const auto* digest = reinterpret_cast<const std::byte*>(
      buffer.data() + sizeof(fsverity_digest));
  return validate_fsverity_sha256_measurement(
      expected.enforced_digest, measurement->digest_algorithm,
      {digest, measurement->digest_size});
#endif
}

}  // namespace

Result<LinuxDeepSeekRankArtifactTransferReceiverOperations>
LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
    std::int32_t control_fd,
    std::uint64_t expected_controller_process_identity) {
  auto status = validate_socket(control_fd,
                                expected_controller_process_identity);
  if (!status.ok()) return status;
  return LinuxDeepSeekRankArtifactTransferReceiverOperations(
      control_fd, expected_controller_process_identity);
}

Status
LinuxDeepSeekRankArtifactTransferReceiverOperations::validate_bound_control(
    std::int32_t control_fd) const {
  if (control_fd != control_fd_) {
    return Status::FailedPrecondition(
        "DeepSeek artifact worker control descriptor changed");
  }
  return validate_socket(control_fd_, controller_process_identity_);
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankArtifactTransferReceiverOperations::receive_manifest(
    std::int32_t control_fd) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  std::array<std::byte, kDeepSeekRankArtifactTransferManifestBytes> frame{};
  auto packet = receive_credentialed_packet(
      control_fd_, frame, controller_process_identity_, false);
  if (!packet.ok()) {
    if (packet.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return packet.status();
  }
  if (packet->bytes != frame.size()) {
    return Status::InvalidArgument(
        "DeepSeek artifact manifest packet size is invalid");
  }
  return std::optional<std::vector<std::byte>>{
      std::in_place, frame.begin(), frame.end()};
}

Result<std::optional<DeepSeekRankArtifactReceivedBatch>>
LinuxDeepSeekRankArtifactTransferReceiverOperations::
    receive_descriptor_batch(std::int32_t control_fd) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  std::array<std::byte,
             kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes>
      frame{};
  auto packet = receive_credentialed_packet(
      control_fd_, frame, controller_process_identity_, true);
  if (!packet.ok()) {
    if (packet.status().code() == StatusCode::kUnavailable) {
      return std::optional<DeepSeekRankArtifactReceivedBatch>{};
    }
    return packet.status();
  }
  auto metadata = decode_deepseek_rank_artifact_descriptor_batch(
      std::span<const std::byte>(frame).first(packet->bytes));
  if (!metadata.ok()) return metadata.status();
  if (packet->descriptors.values.size() !=
      metadata->fields().descriptor_count) {
    return Status::FailedPrecondition(
        "DeepSeek artifact ancillary descriptor count differs");
  }

  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.reserve(packet->descriptors.values.size());
  for (std::size_t index = 0;
       index < packet->descriptors.values.size(); ++index) {
    status = validate_received_descriptor(
        packet->descriptors.values[index], metadata->expectations()[index]);
    if (!status.ok()) return status;
    auto adopted = ArtifactWorkerDescriptor::AdoptLinuxFileDescriptor(
        &packet->descriptors.values[index]);
    if (!adopted.ok()) return adopted.status();
    descriptors.emplace_back(metadata->expectations()[index].shard_name,
                             std::move(*adopted));
  }
  return std::optional<DeepSeekRankArtifactReceivedBatch>{
      std::in_place,
      DeepSeekRankArtifactReceivedBatch{
          std::move(*metadata), std::move(descriptors)}};
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankArtifactTransferReceiverOperations::receive_chunk(
    std::int32_t control_fd) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  std::array<std::byte,
             kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes>
      frame{};
  auto packet = receive_credentialed_packet(
      control_fd_, frame, controller_process_identity_, false);
  if (!packet.ok()) {
    if (packet.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return packet.status();
  }
  auto chunk = decode_deepseek_rank_artifact_metadata_chunk(
      std::span<const std::byte>(frame).first(packet->bytes));
  if (!chunk.ok()) return chunk.status();
  return std::optional<std::vector<std::byte>>{
      std::in_place, frame.begin(),
      frame.begin() + static_cast<std::ptrdiff_t>(packet->bytes)};
}

Result<std::uint64_t>
LinuxDeepSeekRankArtifactTransferReceiverOperations::monotonic_now_ns() {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0 ||
      value.tv_sec < 0 || value.tv_nsec < 0 ||
      value.tv_nsec >= 1'000'000'000L) {
    return system_failure("clock_gettime artifact receiver");
  }
  return static_cast<std::uint64_t>(value.tv_sec) *
             UINT64_C(1'000'000'000) +
         static_cast<std::uint64_t>(value.tv_nsec);
}

Status LinuxDeepSeekRankArtifactTransferReceiverOperations::send_ack(
    std::int32_t control_fd, std::span<const std::byte> frame) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  if (frame.size() != kDeepSeekRankArtifactTransferAckBytes &&
      frame.size() !=
          kDeepSeekRankArtifactMetadataChunkAckFrameBytes) {
    return Status::InvalidArgument(
        "DeepSeek artifact ACK frame size is invalid");
  }
  const auto written = ::send(
      control_fd_, frame.data(), frame.size(),
      MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek artifact ACK socket is backpressured");
  }
  if (written < 0) return system_failure("send artifact ACK");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek artifact ACK was partially sent");
  }
  return Status::Ok();
}

}  // namespace pih
