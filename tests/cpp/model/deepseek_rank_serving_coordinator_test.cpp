#include "pih/model/deepseek_rank_serving_coordinator.h"

#include <deque>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest root() {
  return Sha256Digest::ParseHex(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
      .value();
}

std::vector<DeepSeekRankServingSessionBinding> sessions() {
  return {{7, 8, 2, 0, 101, 201, 301, root()},
          {7, 8, 2, 1, 102, 202, 302, root()}};
}

class Channel final : public DeepSeekRankServingChannel {
 public:
  Status send_command(std::uint32_t rank, std::span<const std::byte> frame) override {
    if (block_first && sent.empty()) return Status::Unavailable("backpressure");
    sent.emplace_back(rank,
                      std::vector<std::byte>(frame.begin(), frame.end()));
    return Status::Ok();
  }
  Result<std::optional<std::vector<std::byte>>> poll_completion(
      std::uint32_t rank) override {
    if (completions[rank].empty()) return std::optional<std::vector<std::byte>>{};
    auto result = std::move(completions[rank].front());
    completions[rank].pop_front();
    return std::optional<std::vector<std::byte>>(std::move(result));
  }
  Status abort_generation(std::uint64_t, std::uint64_t, const Status&) override {
    ++aborts;
    return Status::Ok();
  }
  bool block_first = false;
  std::vector<std::pair<std::uint32_t, std::vector<std::byte>>> sent;
  std::array<std::deque<std::vector<std::byte>>, 2> completions;
  std::uint32_t aborts = 0;
};

DeepSeekRankServingCompletion completion(const DeepSeekRankServingCommand& command) {
  DeepSeekRankServingCompletion value;
  value.session = command.session;
  value.execution_command_sequence = command.command_sequence;
  value.request_id = command.request_id;
  value.request_generation = command.request_generation;
  value.plan = command.plan;
  value.output_lease_identity = command.output_lease_identity;
  return value;
}

std::vector<std::byte> completion_frame(
    const DeepSeekRankServingCommand& command) {
  const auto encoded = encode_deepseek_rank_serving_completion(
      completion(command));
  return {encoded.begin(), encoded.end()};
}

TEST(DeepSeekRankServingCoordinatorTest,
     RetriesExactFramesAndCompletesOnlyAfterEveryRank) {
  Channel channel;
  channel.block_first = true;
  auto coordinator = DeepSeekRankServingCoordinator::Create(sessions(), channel).value();
  ASSERT_TRUE(coordinator.begin(9, 10, {7, 11, DeepSeekPlanPhase::kDecode, 1, 1},
                                12, 13).ok());
  ASSERT_TRUE(coordinator.advance().ok());
  EXPECT_TRUE(channel.sent.empty());
  channel.block_first = false;
  ASSERT_TRUE(coordinator.advance().ok());
  ASSERT_EQ(channel.sent.size(), 2U);
  auto first = decode_deepseek_rank_serving_command(channel.sent[0].second);
  auto last = decode_deepseek_rank_serving_command(channel.sent[1].second);
  ASSERT_TRUE(first.ok()); ASSERT_TRUE(last.ok());
  EXPECT_EQ(first->input_lease_identity, 12U);
  EXPECT_EQ(first->output_lease_identity, 0U);
  EXPECT_EQ(last->input_lease_identity, 0U);
  EXPECT_EQ(last->output_lease_identity, 13U);
  channel.completions[0].push_back(completion_frame(*first));
  ASSERT_TRUE(coordinator.advance().ok());
  EXPECT_TRUE(coordinator.active());
  channel.completions[1].push_back(completion_frame(*last));
  ASSERT_TRUE(coordinator.advance().ok());
  EXPECT_TRUE(coordinator.complete());
  EXPECT_FALSE(coordinator.active());
  EXPECT_EQ(channel.aborts, 0U);
}

TEST(DeepSeekRankServingCoordinatorTest,
     TreatsPostCancellationOutputAsGenerationFailure) {
  Channel channel;
  auto coordinator = DeepSeekRankServingCoordinator::Create(sessions(), channel).value();
  ASSERT_TRUE(coordinator.begin(9, 10, {7, 11, DeepSeekPlanPhase::kDecode, 1, 1},
                                12, 13).ok());
  ASSERT_TRUE(coordinator.advance().ok());
  ASSERT_TRUE(coordinator.cancel(9, 10).ok());
  ASSERT_TRUE(coordinator.advance().ok());
  ASSERT_EQ(channel.sent.size(), 4U);
  auto execute_last = decode_deepseek_rank_serving_command(channel.sent[1].second);
  ASSERT_TRUE(execute_last.ok());
  channel.completions[1].push_back(completion_frame(*execute_last));
  EXPECT_EQ(coordinator.advance().code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(coordinator.poisoned());
  EXPECT_EQ(channel.aborts, 1U);
}

TEST(DeepSeekRankServingCoordinatorTest,
     RejectsInvalidPlanPhaseBeforeAnyRankReceivesAFrame) {
  Channel channel;
  auto coordinator = DeepSeekRankServingCoordinator::Create(sessions(), channel).value();
  auto plan = DeepSeekPipelinePlanDescriptor{
      7, 11, static_cast<DeepSeekPlanPhase>(99), 1, 1};
  EXPECT_EQ(coordinator.begin(9, 10, plan, 12, 13).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(channel.sent.empty());
  EXPECT_FALSE(coordinator.active());
}

TEST(DeepSeekRankServingCoordinatorTest,
     RejectsDrainPlanBeforeAnyRankReceivesAFrame) {
  Channel channel;
  auto coordinator =
      DeepSeekRankServingCoordinator::Create(sessions(), channel).value();
  auto plan = DeepSeekPipelinePlanDescriptor{
      7, 11, DeepSeekPlanPhase::kDrain, 0, 1};
  EXPECT_EQ(coordinator.begin(9, 10, plan, 12, 13).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(channel.sent.empty());
  EXPECT_FALSE(coordinator.active());
}

}  // namespace
}  // namespace pih
