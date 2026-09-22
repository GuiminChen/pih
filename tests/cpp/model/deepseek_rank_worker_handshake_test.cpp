#include "pih/model/deepseek_rank_worker_handshake.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest uuid_commitment() {
  Sha256Digest value{}; value.bytes.fill(std::byte{7}); return value;
}

DeepSeekRankWorkerArguments arguments() {
  return {{7, 8, 2, 1, 10, 11, {}, 5, 200}, 90, 12, 13,
          false, 134217728,
          {"--model=/weights"}};
}

DeepSeekRankExecChallenge challenge() {
  return {1, {7, 8, 2, 1, 10, 11, uuid_commitment(), 5, 200},
          {100, 200, 300}, 90, 400};
}

DeepSeekRankExecObservation observation() {
  return {{7, 8, 2, 1, 10, 11, {}, 5, 200}, 100, 90, 90, 10,
          uuid_commitment(), 5, 9, true, true};
}

class HandshakeOperations final : public DeepSeekRankWorkerHandshakeOperations {
 public:
  Result<std::optional<std::vector<std::byte>>> receive_challenge(
      std::int32_t fd) override {
    ++receives; observed_fd = fd;
    if (pending_receive) return std::optional<std::vector<std::byte>>{};
    auto bytes = encode_deepseek_rank_challenge(challenge_value);
    return std::optional<std::vector<std::byte>>{
        std::vector<std::byte>(bytes.begin(), bytes.end())};
  }
  Result<DeepSeekRankExecObservation> collect_observation(
      const DeepSeekRankWorkerArguments&) override {
    ++collections; return observation_value;
  }
  Status send_ready(std::int32_t fd,
                    std::span<const std::byte> frame) override {
    ++sends; observed_fd = fd;
    if (pending_sends > 0) { --pending_sends; return Status::Unavailable("busy"); }
    sent.assign(frame.begin(), frame.end()); return send_status;
  }
  DeepSeekRankExecChallenge challenge_value = challenge();
  DeepSeekRankExecObservation observation_value = observation();
  Status send_status = Status::Ok();
  bool pending_receive = false;
  int pending_sends = 0;
  int receives = 0;
  int collections = 0;
  int sends = 0;
  int observed_fd = -1;
  std::vector<std::byte> sent;
};

TEST(DeepSeekRankWorkerHandshakeTest, VerifiesAndSendsExactlyOneReadyFrame) {
  HandshakeOperations operations;
  auto handshake = DeepSeekRankWorkerHandshake::Create(
      arguments(), operations).value();
  ASSERT_TRUE(handshake.poll(100).ok());
  EXPECT_TRUE(handshake.ready());
  EXPECT_EQ(handshake.challenge_identity(), 400U);
  EXPECT_EQ(operations.receives, 1);
  EXPECT_EQ(operations.collections, 1);
  EXPECT_EQ(operations.sends, 1);
  EXPECT_EQ(operations.sent.size(), kDeepSeekRankReadyBytes);
  auto ready = decode_deepseek_rank_ready(operations.sent);
  ASSERT_TRUE(ready.ok());
  EXPECT_EQ(ready->receipt.rank, 1U);
}

TEST(DeepSeekRankWorkerHandshakeTest, RetriesBackpressureWithoutReobserving) {
  HandshakeOperations operations;
  operations.pending_sends = 1;
  auto handshake = DeepSeekRankWorkerHandshake::Create(
      arguments(), operations).value();
  EXPECT_EQ(handshake.poll(100).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(handshake.poisoned());
  ASSERT_TRUE(handshake.poll(101).ok());
  EXPECT_TRUE(handshake.ready());
  EXPECT_EQ(operations.receives, 1);
  EXPECT_EQ(operations.collections, 1);
  EXPECT_EQ(operations.sends, 2);
}

TEST(DeepSeekRankWorkerHandshakeTest, PendingChallengeDoesNotPoison) {
  HandshakeOperations operations;
  operations.pending_receive = true;
  auto handshake = DeepSeekRankWorkerHandshake::Create(
      arguments(), operations).value();
  EXPECT_EQ(handshake.poll(100).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(handshake.poisoned());
  operations.pending_receive = false;
  EXPECT_TRUE(handshake.poll(101).ok());
}

TEST(DeepSeekRankWorkerHandshakeTest, IdentityDriftPoisonsWithoutReady) {
  HandshakeOperations operations;
  ++operations.observation_value.actual_parent_process_identity;
  auto handshake = DeepSeekRankWorkerHandshake::Create(
      arguments(), operations).value();
  EXPECT_FALSE(handshake.poll(100).ok());
  EXPECT_TRUE(handshake.poisoned());
  EXPECT_EQ(operations.sends, 0);
  EXPECT_TRUE(operations.sent.empty());
}

TEST(DeepSeekRankWorkerHandshakeTest, DeadlineEqualityPoisonsWithoutReady) {
  HandshakeOperations operations;
  operations.pending_receive = true;
  auto handshake = DeepSeekRankWorkerHandshake::Create(
      arguments(), operations).value();
  EXPECT_EQ(handshake.poll(199).code(), StatusCode::kUnavailable);
  EXPECT_EQ(handshake.poll(200).code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(handshake.poisoned());
  EXPECT_EQ(operations.collections, 0);
  EXPECT_EQ(operations.sends, 0);
}

}  // namespace
}  // namespace pih
