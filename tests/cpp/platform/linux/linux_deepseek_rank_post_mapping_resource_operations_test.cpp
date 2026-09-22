#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_deepseek_rank_post_mapping_resource_operations.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace pih {
namespace {

class CredentialedSocketPair final {
 public:
  explicit CredentialedSocketPair(bool pass_credentials = true) {
    if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0,
                     descriptors_) != 0) {
      descriptors_[0] = descriptors_[1] = -1;
      return;
    }
    if (pass_credentials) {
      const int enabled = 1;
      if (::setsockopt(descriptors_[0], SOL_SOCKET, SO_PASSCRED, &enabled,
                       sizeof(enabled)) != 0 ||
          ::setsockopt(descriptors_[1], SOL_SOCKET, SO_PASSCRED, &enabled,
                       sizeof(enabled)) != 0) {
        close_all();
      }
    }
  }

  ~CredentialedSocketPair() { close_all(); }
  CredentialedSocketPair(const CredentialedSocketPair&) = delete;
  CredentialedSocketPair& operator=(const CredentialedSocketPair&) = delete;

  int first() const noexcept { return descriptors_[0]; }
  int second() const noexcept { return descriptors_[1]; }
  void close_first() noexcept { close_one(0); }
  void close_second() noexcept { close_one(1); }

 private:
  void close_one(std::size_t index) noexcept {
    if (descriptors_[index] >= 0) {
      (void)::close(descriptors_[index]);
      descriptors_[index] = -1;
    }
  }
  void close_all() noexcept {
    close_one(0);
    close_one(1);
  }

  int descriptors_[2]{-1, -1};
};

Sha256Digest post_mapping_operations_digest(std::uint8_t seed) {
  Sha256Digest value{};
  for (std::size_t index = 0; index < value.bytes.size(); ++index) {
    value.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return value;
}

DeepSeekRankPostMappingResourceAuthority make_authority() {
  DeepSeekRankPostMappingResourceAuthority value{};
  value.protocol_version = 1;
  value.engine_epoch = 7;
  value.worker_generation = 8;
  value.world_size = 1;
  value.rank = 0;
  value.process_manifest_identity = 20;
  value.process_identity = 100;
  value.pidfd_identity = 200;
  value.control_identity = 300;
  value.challenge_identity = 400;
  value.manifest_root = post_mapping_operations_digest(1);
  value.spawn_resource_plan_root = post_mapping_operations_digest(2);
  value.capacity_plan_instance_root = post_mapping_operations_digest(3);
  value.os_resource_envelope_root = post_mapping_operations_digest(4);
  value.first_resource_seal_root = post_mapping_operations_digest(5);
  value.first_resource_plan_set_root = post_mapping_operations_digest(6);
  value.descriptor_transaction_root = post_mapping_operations_digest(7);
  value.metadata_transaction_root = post_mapping_operations_digest(8);
  value.metadata_root = post_mapping_operations_digest(9);
  value.expected_mapping_owner_root = post_mapping_operations_digest(10);
  value.deadline_ns = 900;
  value.expected_mapped_interval_bytes = 4096;
  value.expected_immutability_mode = ArtifactImmutabilityMode::kFsVerity;
  value.expected_source_catalog_production_eligible = true;
  value.dspark_enabled = false;
  value.resource_plan = {8, 20, 200, 2, 4, 10,
                         value.os_resource_envelope_root};
  return value;
}

DeepSeekRankPostMappingResourceObservation make_observation() {
  return {{7,
           8,
           0,
           100,
           400,
           post_mapping_operations_digest(1),
           post_mapping_operations_digest(2),
           8,
           20,
           0,
           200,
           24,
           100,
           1000,
           210,
           true},
          post_mapping_operations_digest(3),
          post_mapping_operations_digest(4),
          post_mapping_operations_digest(5),
          post_mapping_operations_digest(6),
          post_mapping_operations_digest(7),
          4096,
          ArtifactImmutabilityMode::kFsVerity,
          true,
          false};
}

std::size_t open_descriptor_count() {
  std::size_t count = 0;
  DIR* directory = ::opendir("/proc/self/fd");
  if (directory == nullptr) return 0;
  while (const auto* entry = ::readdir(directory)) {
    if (std::strcmp(entry->d_name, ".") != 0 &&
        std::strcmp(entry->d_name, "..") != 0) {
      ++count;
    }
  }
  (void)::closedir(directory);
  return count;
}

ssize_t send_with_rights(int socket, std::span<const std::byte> frame,
                         int descriptor) {
  iovec vector{const_cast<std::byte*>(frame.data()), frame.size()};
  alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int))> control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  auto* header = CMSG_FIRSTHDR(&message);
  if (header == nullptr) return -1;
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(header), &descriptor, sizeof(descriptor));
  return ::sendmsg(socket, &message, MSG_NOSIGNAL);
}

TEST(LinuxDeepSeekRankPostMappingResourceOperationsTest,
     ExchangesExactFramesAcrossProcessesWithKernelCredentials) {
  CredentialedSocketPair sockets;
  ASSERT_GE(sockets.first(), 0);
  auto authority =
      encode_deepseek_rank_post_mapping_resource_authority(make_authority());
  auto observation =
      encode_deepseek_rank_post_mapping_resource_observation(
          make_observation());
  ASSERT_TRUE(authority.ok()) << authority.status().message();
  ASSERT_TRUE(observation.ok()) << observation.status().message();
  ASSERT_EQ(::send(sockets.first(), authority->data(), authority->size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(authority->size()));

  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    sockets.close_first();
    auto operations =
        LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
            sockets.second(), static_cast<std::uint64_t>(::getppid()));
    if (!operations.ok()) _exit(10);
    auto received = operations->receive_authority(sockets.second());
    if (!received.ok() || !received->has_value() ||
        received->value() != std::vector<std::byte>(
                                 authority->begin(), authority->end())) {
      _exit(11);
    }
    if (!operations->send_observation(sockets.second(), *observation).ok()) {
      _exit(12);
    }
    _exit(0);
  }

  sockets.close_second();
  std::array<std::byte,
             kDeepSeekRankPostMappingResourceObservationFrameBytes>
      received{};
  iovec vector{received.data(), received.size()};
  alignas(cmsghdr) std::array<std::byte,
      CMSG_SPACE(sizeof(struct ucred))> control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  ASSERT_EQ(::recvmsg(sockets.first(), &message, 0),
            static_cast<ssize_t>(received.size()));
  ASSERT_EQ(message.msg_flags & (MSG_TRUNC | MSG_CTRUNC), 0);
  auto* credentials = CMSG_FIRSTHDR(&message);
  ASSERT_NE(credentials, nullptr);
  ASSERT_EQ(credentials->cmsg_level, SOL_SOCKET);
  ASSERT_EQ(credentials->cmsg_type, SCM_CREDENTIALS);
  ASSERT_EQ(credentials->cmsg_len, CMSG_LEN(sizeof(struct ucred)));
  struct ucred observed_identity {};
  std::memcpy(&observed_identity, CMSG_DATA(credentials),
              sizeof(observed_identity));
  EXPECT_EQ(observed_identity.pid, child);
  EXPECT_EQ(observed_identity.uid, ::geteuid());
  EXPECT_EQ(observed_identity.gid, ::getegid());
  EXPECT_EQ(CMSG_NXTHDR(&message, credentials), nullptr);
  EXPECT_EQ(received, *observation);

  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
  EXPECT_EQ(
      kLinuxDeepSeekRankPostMappingResourceReporterOperationsAbi,
      "pih_linux_deepseek_rank_post_mapping_resource_reporter_operations_v1");
}

TEST(LinuxDeepSeekRankPostMappingResourceOperationsTest,
     RejectsRightsAndClosesEveryReceivedDescriptor) {
  CredentialedSocketPair sockets;
  ASSERT_GE(sockets.first(), 0);
  auto operations =
      LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok()) << operations.status().message();
  auto authority =
      encode_deepseek_rank_post_mapping_resource_authority(make_authority());
  ASSERT_TRUE(authority.ok());
  const int descriptor = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  const auto before = open_descriptor_count();
  ASSERT_NE(before, 0U);
  ASSERT_EQ(send_with_rights(sockets.first(), *authority, descriptor),
            static_cast<ssize_t>(authority->size()));
  auto received = operations->receive_authority(sockets.second());
  EXPECT_FALSE(received.ok());
  EXPECT_EQ(received.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(open_descriptor_count(), before);
  (void)::close(descriptor);
}

TEST(LinuxDeepSeekRankPostMappingResourceOperationsTest,
     RequiresPasscredPeerAndExactBoundPacket) {
  CredentialedSocketPair without_credentials(false);
  EXPECT_FALSE(
      LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
          without_credentials.second(),
          static_cast<std::uint64_t>(::getpid()))
          .ok());

  CredentialedSocketPair sockets;
  EXPECT_FALSE(
      LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()) + 1U)
          .ok());
  auto operations =
      LinuxDeepSeekRankPostMappingResourceReporterOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok());
  EXPECT_FALSE(operations->receive_authority(sockets.first()).ok());

  auto authority =
      encode_deepseek_rank_post_mapping_resource_authority(make_authority());
  ASSERT_TRUE(authority.ok());
  ASSERT_EQ(::send(sockets.first(), authority->data(), authority->size() - 1U,
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(authority->size() - 1U));
  auto short_packet = operations->receive_authority(sockets.second());
  EXPECT_FALSE(short_packet.ok());
  EXPECT_EQ(short_packet.status().code(), StatusCode::kFailedPrecondition);

  std::vector<std::byte> oversized(authority->begin(), authority->end());
  oversized.push_back(std::byte{0});
  ASSERT_EQ(::send(sockets.first(), oversized.data(), oversized.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(oversized.size()));
  auto long_packet = operations->receive_authority(sockets.second());
  EXPECT_FALSE(long_packet.ok());
  EXPECT_EQ(long_packet.status().code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
