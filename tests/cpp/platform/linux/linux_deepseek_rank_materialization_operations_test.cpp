#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_deepseek_rank_materialization_operations.h"

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

Sha256Digest materialization_operations_digest(std::uint8_t seed) {
  Sha256Digest value{};
  for (std::size_t index = 0; index < value.bytes.size(); ++index) {
    value.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return value;
}

DeepSeekRankMaterializationGrantFields make_grant() {
  return {2,
          7,
          8,
          1,
          0,
          20,
          100,
          200,
          300,
          400,
          0,
          RuntimeProfileGpuFamily::kH100Pcie80GiB,
          RuntimeProfileResidency::kHostSpill,
          true,
          false,
          900,
          materialization_operations_digest(1),
          materialization_operations_digest(2),
          materialization_operations_digest(3),
          materialization_operations_digest(4),
          materialization_operations_digest(5),
          materialization_operations_digest(6),
          materialization_operations_digest(7),
          materialization_operations_digest(8),
          materialization_operations_digest(9),
          materialization_operations_digest(10),
          materialization_operations_digest(11),
          materialization_operations_digest(12),
          materialization_operations_digest(13),
          materialization_operations_digest(14)};
}

DeepSeekRankMaterializationGrantAckFields make_ack(
    const DeepSeekRankMaterializationGrantFields& grant) {
  const auto grant_root =
      compile_deepseek_rank_materialization_grant_root(grant).value();
  return {1,
          grant.engine_epoch,
          grant.worker_generation,
          grant.world_size,
          grant.rank,
          grant.process_manifest_identity,
          grant.process_identity,
          grant.pidfd_identity,
          grant.control_identity,
          grant.challenge_identity,
          grant_root,
          grant.report_root,
          grant.mapping_owner_root,
          grant.post_mapping_seal_root};
}

DeepSeekRankMaterializationCompletionFields make_completion(
    const DeepSeekRankMaterializationGrantFields& grant) {
  return {1,
          grant.engine_epoch,
          grant.worker_generation,
          grant.world_size,
          grant.rank,
          grant.process_manifest_identity,
          grant.process_identity,
          grant.pidfd_identity,
          grant.control_identity,
          grant.challenge_identity,
          grant.device_ordinal,
          grant.gpu_family,
          grant.residency,
          grant.production_eligible,
          grant.dspark_enabled,
          700,
          800,
          grant.deadline_ns,
          4097,
          8192,
          8192,
          2,
          2,
          8192,
          4096,
          101,
          2 * 13'369'344ULL,
          202,
          2,
          2,
          0,
          grant.profile_envelope_root,
          grant.device_observation_root,
          grant.capacity_plan_instance_root,
          grant.post_mapping_seal_root,
          grant.metadata_transaction_root,
          grant.mapping_owner_root,
          compile_deepseek_rank_materialization_grant_root(grant).value(),
          materialization_operations_digest(20),
          materialization_operations_digest(21),
          materialization_operations_digest(22),
          materialization_operations_digest(23),
          materialization_operations_digest(24),
          materialization_operations_digest(25)};
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

TEST(LinuxDeepSeekRankMaterializationOperationsTest,
     ExchangesExactFramesAcrossProcessesWithKernelCredentials) {
  CredentialedSocketPair sockets;
  ASSERT_GE(sockets.first(), 0);
  const auto grant_fields = make_grant();
  auto grant = encode_deepseek_rank_materialization_grant(grant_fields);
  auto ack = encode_deepseek_rank_materialization_grant_ack(
      make_ack(grant_fields));
  auto completion = encode_deepseek_rank_materialization_completion(
      make_completion(grant_fields));
  ASSERT_TRUE(grant.ok()) << grant.status().message();
  ASSERT_TRUE(ack.ok()) << ack.status().message();
  ASSERT_TRUE(completion.ok()) << completion.status().message();
  ASSERT_EQ(::send(sockets.first(), grant->data(), grant->size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(grant->size()));

  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    sockets.close_first();
    auto operations =
        LinuxDeepSeekRankMaterializationReceiverOperations::Create(
            sockets.second(), static_cast<std::uint64_t>(::getppid()));
    if (!operations.ok()) _exit(10);
    auto received = operations->receive_grant(sockets.second());
    if (!received.ok() || !received->has_value() ||
        received->value() != std::vector<std::byte>(
                                 grant->begin(), grant->end())) {
      _exit(11);
    }
    if (!operations->send_ack(sockets.second(), *ack).ok()) _exit(12);
    if (!operations->send_completion(sockets.second(), *completion).ok()) {
      _exit(13);
    }
    _exit(0);
  }

  sockets.close_second();
  std::array<std::byte,
             kDeepSeekRankMaterializationGrantAckFrameBytes> received{};
  iovec vector{received.data(), received.size()};
  alignas(cmsghdr) std::array<
      std::byte, CMSG_SPACE(sizeof(struct ucred))> control{};
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
  EXPECT_EQ(received, *ack);

  std::array<std::byte,
             kDeepSeekRankMaterializationCompletionFrameBytes>
      received_completion{};
  iovec completion_vector{received_completion.data(),
                           received_completion.size()};
  control.fill(std::byte{0});
  message = {};
  message.msg_iov = &completion_vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  ASSERT_EQ(::recvmsg(sockets.first(), &message, 0),
            static_cast<ssize_t>(received_completion.size()));
  EXPECT_EQ(received_completion, *completion);
  EXPECT_TRUE(decode_deepseek_rank_materialization_completion(
                  received_completion)
                  .ok());

  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
  EXPECT_EQ(
      kLinuxDeepSeekRankMaterializationReceiverOperationsAbi,
      "pih_linux_deepseek_rank_materialization_receiver_operations_v1");
}

TEST(LinuxDeepSeekRankMaterializationOperationsTest,
     RejectsRightsAndClosesEveryReceivedDescriptor) {
  CredentialedSocketPair sockets;
  ASSERT_GE(sockets.first(), 0);
  auto operations =
      LinuxDeepSeekRankMaterializationReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok()) << operations.status().message();
  auto grant = encode_deepseek_rank_materialization_grant(make_grant());
  ASSERT_TRUE(grant.ok());
  const int descriptor = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  const auto before = open_descriptor_count();
  ASSERT_NE(before, 0U);
  ASSERT_EQ(send_with_rights(sockets.first(), *grant, descriptor),
            static_cast<ssize_t>(grant->size()));
  auto received = operations->receive_grant(sockets.second());
  EXPECT_FALSE(received.ok());
  EXPECT_EQ(received.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(open_descriptor_count(), before);
  (void)::close(descriptor);
}

TEST(LinuxDeepSeekRankMaterializationOperationsTest,
     RequiresPasscredPeerAndExactBoundPacket) {
  CredentialedSocketPair without_credentials(false);
  EXPECT_FALSE(
      LinuxDeepSeekRankMaterializationReceiverOperations::Create(
          without_credentials.second(),
          static_cast<std::uint64_t>(::getpid()))
          .ok());

  CredentialedSocketPair sockets;
  EXPECT_FALSE(
      LinuxDeepSeekRankMaterializationReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()) + 1U)
          .ok());
  auto operations =
      LinuxDeepSeekRankMaterializationReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok());
  EXPECT_FALSE(operations->receive_grant(sockets.first()).ok());

  auto grant = encode_deepseek_rank_materialization_grant(make_grant());
  ASSERT_TRUE(grant.ok());
  ASSERT_EQ(::send(sockets.first(), grant->data(), grant->size() - 1U,
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(grant->size() - 1U));
  auto short_packet = operations->receive_grant(sockets.second());
  EXPECT_FALSE(short_packet.ok());
  EXPECT_EQ(short_packet.status().code(), StatusCode::kFailedPrecondition);

  std::vector<std::byte> oversized(grant->begin(), grant->end());
  oversized.push_back(std::byte{0});
  ASSERT_EQ(::send(sockets.first(), oversized.data(), oversized.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(oversized.size()));
  auto long_packet = operations->receive_grant(sockets.second());
  EXPECT_FALSE(long_packet.ok());
  EXPECT_EQ(long_packet.status().code(), StatusCode::kFailedPrecondition);
}

TEST(LinuxDeepSeekRankMaterializationOperationsTest,
     ExchangesServingFramesAcrossProcessesWithKernelCredentials) {
  CredentialedSocketPair sockets;
  ASSERT_GE(sockets.first(), 0);
  auto seal = materialization_operations_digest(44);
  DeepSeekRankServingCommand command;
  command.session = {7, 8, 1, 0, 100, 200, 300, seal};
  command.command_sequence = 9;
  command.request_id = 10;
  command.request_generation = 11;
  command.plan = {7, 12, DeepSeekPlanPhase::kDecode, 1, 1};
  command.input_lease_identity = 13;
  command.output_lease_identity = 14;
  const auto command_frame = encode_deepseek_rank_serving_command(command);
  ASSERT_EQ(::send(sockets.first(), command_frame.data(), command_frame.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(command_frame.size()));

  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    sockets.close_first();
    auto operations = LinuxDeepSeekRankServingWorkerOperations::Create(
        sockets.second(), static_cast<std::uint64_t>(::getppid()));
    if (!operations.ok()) _exit(20);
    auto received = operations->receive_command();
    if (!received.ok() || !received->has_value() ||
        received->value() != std::vector<std::byte>(command_frame.begin(),
                                                     command_frame.end())) {
      _exit(21);
    }
    DeepSeekRankServingCompletion completion;
    completion.session = command.session;
    completion.execution_command_sequence = command.command_sequence;
    completion.request_id = command.request_id;
    completion.request_generation = command.request_generation;
    completion.plan = command.plan;
    completion.output_lease_identity = command.output_lease_identity;
    const auto completion_frame =
        encode_deepseek_rank_serving_completion(completion);
    if (!operations->send_completion(completion_frame).ok()) _exit(22);
    _exit(0);
  }

  sockets.close_second();
  std::array<std::byte, kDeepSeekRankServingCompletionBytes> received{};
  iovec vector{received.data(), received.size()};
  alignas(cmsghdr) std::array<
      std::byte, CMSG_SPACE(sizeof(struct ucred))> control{};
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
  struct ucred observed{};
  std::memcpy(&observed, CMSG_DATA(credentials), sizeof(observed));
  EXPECT_EQ(observed.pid, child);
  EXPECT_TRUE(decode_deepseek_rank_serving_completion(received).ok());

  int child_status = 0;
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
  EXPECT_EQ(kLinuxDeepSeekRankServingWorkerOperationsAbi,
            "pih_linux_deepseek_rank_serving_worker_operations_v1");
}

class ImmediateServingExecution final : public DeepSeekRankServingPlanExecution {
 public:
  Status cancel() override {
    cancelled = true;
    return Status::Ok();
  }
  Result<bool> advance() override {
    ++advances;
    return true;
  }

  bool cancelled = false;
  std::uint32_t advances = 0;
};

class ImmediateServingFactory final
    : public DeepSeekRankServingPlanExecutionFactory {
 public:
  Result<std::unique_ptr<DeepSeekRankServingPlanExecution>> start(
      const DeepSeekRankServingCommand& command,
      DeepSeekRankServingResolvedLeases) override {
    received = command;
    auto execution = std::make_unique<ImmediateServingExecution>();
    last = execution.get();
    return std::unique_ptr<DeepSeekRankServingPlanExecution>(
        std::move(execution));
  }

  DeepSeekRankServingCommand received;
  ImmediateServingExecution* last = nullptr;
};

class ImmediateServingResolver final : public DeepSeekRankServingLeaseResolver {
 public:
  Result<DeepSeekRankServingResolvedLeases> resolve(
      const DeepSeekRankServingCommand& command) override {
    DeepSeekRankServingResolvedLeases leases;
    leases.input_lease_identity = command.input_lease_identity;
    leases.output_lease_identity = command.output_lease_identity;
    if (command.input_lease_identity != 0) leases.input_owner = std::make_shared<int>(1);
    if (command.output_lease_identity != 0) leases.output_owner = std::make_shared<int>(2);
    return leases;
  }
};

TEST(LinuxDeepSeekRankMaterializationOperationsTest,
     WorkerOwnsAuthenticatedServingGraphFromCommandToCompletion) {
  CredentialedSocketPair sockets;
  ASSERT_GE(sockets.first(), 0);
  const auto seal = materialization_operations_digest(45);
  const DeepSeekRankServingSessionBinding session{
      7, 8, 1, 0, 100, 200, 300, seal};
  DeepSeekRankServingCommand command;
  command.session = session;
  command.command_sequence = 9;
  command.request_id = 10;
  command.request_generation = 11;
  command.plan = {7, 12, DeepSeekPlanPhase::kDecode, 1, 1};
  command.input_lease_identity = 13;
  command.output_lease_identity = 14;
  const auto command_frame = encode_deepseek_rank_serving_command(command);
  ASSERT_EQ(::send(sockets.first(), command_frame.data(), command_frame.size(),
                   MSG_NOSIGNAL),
            static_cast<ssize_t>(command_frame.size()));

  ImmediateServingFactory factory;
  ImmediateServingResolver resolver;
  auto worker = LinuxDeepSeekRankServingWorker::Create(
      session, sockets.second(), static_cast<std::uint64_t>(::getpid()),
      factory, resolver);
  ASSERT_TRUE(worker.ok()) << worker.status().message();
  ASSERT_TRUE(worker->advance().ok());
  ASSERT_NE(factory.last, nullptr);
  EXPECT_EQ(factory.received.request_id, command.request_id);
  EXPECT_EQ(factory.last->advances, 1U);
  ASSERT_TRUE(worker->advance().ok());

  std::array<std::byte, kDeepSeekRankServingCompletionBytes> received{};
  ASSERT_EQ(::recv(sockets.first(), received.data(), received.size(), 0),
            static_cast<ssize_t>(received.size()));
  auto completion = decode_deepseek_rank_serving_completion(received);
  ASSERT_TRUE(completion.ok()) << completion.status().message();
  EXPECT_EQ(completion->session, session);
  EXPECT_EQ(completion->execution_command_sequence,
            command.command_sequence);
  EXPECT_EQ(completion->output_lease_identity,
            command.output_lease_identity);
  EXPECT_FALSE(worker->failed());
}

}  // namespace
}  // namespace pih
