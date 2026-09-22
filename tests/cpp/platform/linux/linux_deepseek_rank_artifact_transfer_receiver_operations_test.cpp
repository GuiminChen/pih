#include "pih/platform/linux/linux_deepseek_rank_artifact_transfer_receiver_operations.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pih/model/deepseek_rank_artifact_descriptor_batch.h"
#include "pih/model/deepseek_rank_artifact_metadata_chunk.h"
#include "pih/core/sha256.h"

namespace pih {
namespace {

class SocketPair final {
 public:
  explicit SocketPair(bool pass_credentials = true) {
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
  ~SocketPair() { close_all(); }
  int first() const { return descriptors_[0]; }
  int second() const { return descriptors_[1]; }

 private:
  void close_all() noexcept {
    for (auto& descriptor : descriptors_) {
      if (descriptor >= 0) (void)::close(descriptor);
      descriptor = -1;
    }
  }
  int descriptors_[2]{-1, -1};
};

class ReadOnlyFixtureFile final {
 public:
  ReadOnlyFixtureFile() {
    char path[] = "/tmp/pih-artifact-transfer-XXXXXX";
    const int writer = ::mkstemp(path);
    if (writer < 0) return;
    const std::array payload{'a', 'b', 'c', 'd'};
    if (::write(writer, payload.data(), payload.size()) !=
            static_cast<ssize_t>(payload.size()) ||
        ::fsync(writer) != 0) {
      (void)::close(writer);
      (void)::unlink(path);
      return;
    }
    (void)::close(writer);
    descriptor_ = ::open(path, O_RDONLY | O_CLOEXEC);
    (void)::unlink(path);
  }
  ~ReadOnlyFixtureFile() {
    if (descriptor_ >= 0) (void)::close(descriptor_);
  }
  int descriptor() const { return descriptor_; }

  ArtifactFileIdentity identity() const {
    struct stat observation {};
    if (::fstat(descriptor_, &observation) != 0) return {};
    return {static_cast<std::uint64_t>(observation.st_size),
            static_cast<std::uint64_t>(observation.st_dev),
            static_cast<std::uint64_t>(observation.st_ino),
            observation.st_mtim.tv_sec,
            static_cast<std::uint32_t>(observation.st_mtim.tv_nsec)};
  }

 private:
  int descriptor_ = -1;
};

Sha256Digest digest(std::uint8_t seed) {
  Sha256Digest result{};
  for (std::size_t index = 0; index < result.bytes.size(); ++index) {
    result.bytes[index] = static_cast<std::byte>(seed + index);
  }
  return result;
}

DeepSeekRankArtifactDescriptorBatch make_batch(
    const ArtifactFileIdentity& identity) {
  std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations{
      {0, "model.safetensors", identity,
       ArtifactImmutabilityMode::kUncalibrated, {}}};
  const auto batch_root =
      compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
          0, 0, expectations)
          .value();
  const auto adopted_root =
      compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
          0, expectations)
          .value();
  return DeepSeekRankArtifactDescriptorBatch::Create(
             {7, 8, 1, 0, 0, 1, 0, 1, 1, true,
              10, 100, 200, 300, 400,
              digest(1), digest(2), digest(3), batch_root, adopted_root},
             std::move(expectations))
      .value();
}

DeepSeekRankArtifactMetadataChunk make_metadata_chunk() {
  std::vector<std::byte> payload{
      std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  const auto payload_digest = sha256(payload).value();
  return DeepSeekRankArtifactMetadataChunk::Create(
             {7, 8, 1, 0, 0, 1, 0, 4, 4,
              10, 100, 200, 300, 400,
              digest(1), digest(2), digest(3), digest(4), digest(5),
              payload_digest},
             std::move(payload))
      .value();
}

ssize_t send_descriptors(int socket, std::span<const std::byte> payload,
                         std::span<const int> descriptors) {
  iovec vector{const_cast<std::byte*>(payload.data()), payload.size()};
  std::vector<std::byte> control(CMSG_SPACE(
      sizeof(int) * descriptors.size()));
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  auto* header = CMSG_FIRSTHDR(&message);
  if (header == nullptr) return -1;
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(int) * descriptors.size());
  std::memcpy(CMSG_DATA(header), descriptors.data(),
              sizeof(int) * descriptors.size());
  return ::sendmsg(socket, &message, MSG_NOSIGNAL);
}

TEST(LinuxDeepSeekRankArtifactTransferReceiverOperationsTest,
     ReceivesCredentialedExactDescriptorPacketWithCloexecAdoption) {
  SocketPair sockets;
  ReadOnlyFixtureFile file;
  ASSERT_GE(sockets.first(), 0);
  ASSERT_GE(file.descriptor(), 0);
  auto operations =
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok()) << operations.status().message();
  EXPECT_EQ(kLinuxDeepSeekRankArtifactTransferReceiverOperationsAbi,
            "pih_linux_deepseek_rank_artifact_transfer_receiver_operations_v1");
  auto batch = make_batch(file.identity());
  auto payload = encode_deepseek_rank_artifact_descriptor_batch(batch);
  ASSERT_TRUE(payload.ok()) << payload.status().message();
  const std::array descriptors{file.descriptor()};
  ASSERT_EQ(send_descriptors(sockets.first(), *payload, descriptors),
            static_cast<ssize_t>(payload->size()));

  auto received = operations->receive_descriptor_batch(sockets.second());
  ASSERT_TRUE(received.ok()) << received.status().message();
  ASSERT_TRUE(received->has_value());
  ASSERT_EQ(received->value().descriptors.size(), 1U);
  EXPECT_EQ(received->value().metadata.frame_root(), batch.frame_root());
  std::array<std::byte, 4> bytes{};
  EXPECT_TRUE(received->value().descriptors[0].descriptor
                  .read_exact(0, bytes)
                  .ok());
  EXPECT_EQ(bytes[2], static_cast<std::byte>('c'));
}

TEST(LinuxDeepSeekRankArtifactTransferReceiverOperationsTest,
     RejectsAncillaryCountMismatchAndTruncatedPayload) {
  SocketPair sockets;
  ReadOnlyFixtureFile first;
  ReadOnlyFixtureFile second;
  auto operations =
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok());
  auto batch = make_batch(first.identity());
  auto payload = encode_deepseek_rank_artifact_descriptor_batch(batch);
  ASSERT_TRUE(payload.ok());
  const std::array descriptors{first.descriptor(), second.descriptor()};
  ASSERT_EQ(send_descriptors(sockets.first(), *payload, descriptors),
            static_cast<ssize_t>(payload->size()));
  EXPECT_FALSE(
      operations->receive_descriptor_batch(sockets.second()).ok());

  auto oversized = *payload;
  oversized.resize(
      kDeepSeekRankArtifactDescriptorBatchFrameMaximumBytes + 1,
      std::byte{0});
  const std::array one_descriptor{first.descriptor()};
  ASSERT_EQ(send_descriptors(sockets.first(), oversized, one_descriptor),
            static_cast<ssize_t>(oversized.size()));
  EXPECT_FALSE(
      operations->receive_descriptor_batch(sockets.second()).ok());
}

TEST(LinuxDeepSeekRankArtifactTransferReceiverOperationsTest,
     RequiresSeqpacketPasscredAndExactBoundDescriptor) {
  SocketPair no_credentials(false);
  EXPECT_FALSE(
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          no_credentials.second(), static_cast<std::uint64_t>(::getpid()))
          .ok());
  SocketPair credentialed;
  EXPECT_FALSE(
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          credentialed.second(),
          static_cast<std::uint64_t>(::getpid()) + 1U)
          .ok());
  auto operations =
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          credentialed.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok());
  EXPECT_FALSE(operations->receive_manifest(credentialed.first()).ok());
}

TEST(LinuxDeepSeekRankArtifactTransferReceiverOperationsTest,
     ReceivesCredentialedMetadataAndSendsExactAckWithoutRights) {
  SocketPair sockets;
  auto operations =
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok()) << operations.status().message();
  auto chunk = make_metadata_chunk();
  auto frame = encode_deepseek_rank_artifact_metadata_chunk(chunk).value();
  ASSERT_EQ(::send(sockets.first(), frame.data(), frame.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(frame.size()));
  auto received = operations->receive_chunk(sockets.second());
  ASSERT_TRUE(received.ok()) << received.status().message();
  ASSERT_TRUE(received->has_value());
  EXPECT_EQ(received->value(), frame);

  const auto& fields = chunk.fields();
  auto ack = DeepSeekRankArtifactMetadataChunkAck::Create(
                 {fields.engine_epoch,
                  fields.worker_generation,
                  fields.world_size,
                  fields.rank,
                  fields.chunk_index,
                  fields.chunk_count,
                  fields.payload_bytes,
                  fields.total_blob_bytes,
                  fields.process_manifest_identity,
                  fields.process_identity,
                  fields.pidfd_identity,
                  fields.control_identity,
                  fields.challenge_identity,
                  fields.transfer_manifest_root,
                  fields.descriptor_transfer_transaction_root,
                  fields.metadata_root,
                  fields.metadata_transaction_root,
                  fields.blob_sha256,
                  fields.chunk_sha256})
                 .value();
  auto ack_frame = encode_deepseek_rank_artifact_metadata_chunk_ack(ack);
  EXPECT_TRUE(operations->send_ack(sockets.second(), ack_frame).ok());
  std::array<std::byte,
             kDeepSeekRankArtifactMetadataChunkAckFrameBytes> observed{};
  EXPECT_EQ(::recv(sockets.first(), observed.data(), observed.size(), 0),
            static_cast<ssize_t>(observed.size()));
  EXPECT_EQ(observed, ack_frame);
}

TEST(LinuxDeepSeekRankArtifactTransferReceiverOperationsTest,
     RejectsRightsAndOversizeOnMetadataChannel) {
  SocketPair sockets;
  ReadOnlyFixtureFile file;
  auto operations =
      LinuxDeepSeekRankArtifactTransferReceiverOperations::Create(
          sockets.second(), static_cast<std::uint64_t>(::getpid()));
  ASSERT_TRUE(operations.ok());
  auto frame = encode_deepseek_rank_artifact_metadata_chunk(
                   make_metadata_chunk())
                   .value();
  const std::array descriptor{file.descriptor()};
  ASSERT_EQ(send_descriptors(sockets.first(), frame, descriptor),
            static_cast<ssize_t>(frame.size()));
  EXPECT_FALSE(operations->receive_chunk(sockets.second()).ok());

  frame.resize(
      kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes + 1U,
      std::byte{0});
  ASSERT_EQ(::send(sockets.first(), frame.data(), frame.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(frame.size()));
  EXPECT_FALSE(operations->receive_chunk(sockets.second()).ok());
}

}  // namespace
}  // namespace pih
