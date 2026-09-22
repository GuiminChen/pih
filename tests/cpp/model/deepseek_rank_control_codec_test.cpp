#include "pih/model/deepseek_rank_control_codec.h"

#include <span>

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekRankServingSessionBinding session() {
  auto root = Sha256Digest::ParseHex(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
  return {7, 8, 2, 1, 101, 102, 103, root.value()};
}

DeepSeekRankServingCommand command() {
  DeepSeekRankServingCommand value;
  value.session = session();
  value.command_sequence = 11;
  value.request_id = 12;
  value.request_generation = 13;
  value.plan = {7, 14, DeepSeekPlanPhase::kDecode, 15, 16};
  value.output_lease_identity = 17;
  return value;
}

TEST(DeepSeekRankControlCodecTest, RoundTripsServingCommandAndCompletion) {
  const auto expected_command = command();
  const auto command_frame = encode_deepseek_rank_serving_command(expected_command);
  EXPECT_EQ(command_frame.size(), kDeepSeekRankServingCommandBytes);
  auto decoded_command = decode_deepseek_rank_serving_command(command_frame);
  ASSERT_TRUE(decoded_command.ok()) << decoded_command.status().message();
  EXPECT_EQ(decoded_command->session, expected_command.session);
  EXPECT_EQ(decoded_command->command_sequence,
            expected_command.command_sequence);
  EXPECT_EQ(decoded_command->plan.plan_sequence,
            expected_command.plan.plan_sequence);
  EXPECT_EQ(decoded_command->output_lease_identity,
            expected_command.output_lease_identity);

  DeepSeekRankServingCompletion expected_completion;
  expected_completion.session = expected_command.session;
  expected_completion.execution_command_sequence =
      expected_command.command_sequence;
  expected_completion.request_id = expected_command.request_id;
  expected_completion.request_generation = expected_command.request_generation;
  expected_completion.plan = expected_command.plan;
  expected_completion.output_lease_identity =
      expected_command.output_lease_identity;
  const auto completion_frame =
      encode_deepseek_rank_serving_completion(expected_completion);
  EXPECT_EQ(completion_frame.size(), kDeepSeekRankServingCompletionBytes);
  auto decoded_completion =
      decode_deepseek_rank_serving_completion(completion_frame);
  ASSERT_TRUE(decoded_completion.ok()) << decoded_completion.status().message();
  EXPECT_EQ(decoded_completion->session, expected_completion.session);
  EXPECT_EQ(decoded_completion->execution_command_sequence,
            expected_completion.execution_command_sequence);
  EXPECT_EQ(decoded_completion->output_lease_identity,
            expected_completion.output_lease_identity);
}

TEST(DeepSeekRankControlCodecTest, RejectsInvalidServingEnumsAndFrameKinds) {
  auto command_frame = encode_deepseek_rank_serving_command(command());
  command_frame[96] = std::byte{9};
  EXPECT_EQ(decode_deepseek_rank_serving_command(command_frame).status().code(),
            StatusCode::kInvalidArgument);

  DeepSeekRankServingCompletion completion;
  completion.session = session();
  completion.execution_command_sequence = 11;
  completion.request_id = 12;
  completion.request_generation = 13;
  completion.plan = {7, 14, DeepSeekPlanPhase::kDecode, 15, 16};
  auto completion_frame = encode_deepseek_rank_serving_completion(completion);
  completion_frame[140] = std::byte{9};
  EXPECT_EQ(
      decode_deepseek_rank_serving_completion(completion_frame).status().code(),
      StatusCode::kInvalidArgument);

  command_frame = encode_deepseek_rank_serving_command(command());
  command_frame[4] = std::byte{6};
  EXPECT_EQ(decode_deepseek_rank_serving_command(command_frame).status().code(),
            StatusCode::kInvalidArgument);

  command_frame = encode_deepseek_rank_serving_command(command());
  command_frame[6] = std::byte{2};
  EXPECT_EQ(decode_deepseek_rank_serving_command(command_frame).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(decode_deepseek_rank_serving_command(
                std::span<const std::byte>(command_frame.data(),
                                           command_frame.size() - 1U))
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  command_frame = encode_deepseek_rank_serving_command(command());
  command_frame[132] = std::byte{9};
  EXPECT_EQ(decode_deepseek_rank_serving_command(command_frame).status().code(),
            StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
