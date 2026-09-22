#include "pih/model/packed_token_plan.h"

#include <array>
#include <limits>

#include <gtest/gtest.h>

namespace pih {
namespace {

PackedSequenceInput input(std::uint64_t generation,
                          std::uint64_t start,
                          std::uint32_t count,
                          std::uint64_t state_generation) {
  const auto identity = std::to_string(generation);
  return {generation, start, count, state_generation,
          sha256(std::as_bytes(std::span(identity))).value()};
}

Sha256Digest resource_root() {
  constexpr std::string_view identity = "resource-vector";
  return sha256(std::as_bytes(std::span(identity))).value();
}

TEST(PackedTokenPlanTest, FreezesUnevenRealTokenPrefixSums) {
  const std::array sequences{
      input(11, 0, 3, 101), input(12, 7, 1, 102), input(13, 9, 4, 103)};
  auto plan = PackedTokenPlan::Create(
      7, 19, PackedTokenPhase::kPrefill, "profile-r1", resource_root(), sequences,
      8, PackedTokenPlanLimits{4, 16, 32});
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->epoch(), 7U);
  EXPECT_EQ(plan->plan_sequence(), 19U);
  EXPECT_EQ(plan->phase(), PackedTokenPhase::kPrefill);
  EXPECT_EQ(plan->sequence_count(), 3U);
  EXPECT_EQ(plan->total_real_tokens(), 8U);
  EXPECT_EQ(plan->execution_bucket_tokens(), 8U);
  EXPECT_EQ(plan->packed_offsets(), (std::vector<std::uint32_t>{0, 3, 4, 8}));
  EXPECT_EQ(plan->ordered_sequence_generations(),
            (std::vector<std::uint64_t>{11, 12, 13}));
  EXPECT_EQ(plan->committed_start_positions(),
            (std::vector<std::uint64_t>{0, 7, 9}));
  EXPECT_EQ(plan->state_generations(),
            (std::vector<std::uint64_t>{101, 102, 103}));
}

TEST(PackedTokenPlanTest, AcceptsExecutionPaddingWithoutChargingRealTokens) {
  const std::array sequences{input(1, 0, 1, 4), input(2, 8, 1, 5)};
  auto plan = PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", resource_root(), sequences, 4,
      PackedTokenPlanLimits{2, 2, 4});
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan->total_real_tokens(), 2U);
  EXPECT_EQ(plan->execution_bucket_tokens(), 4U);
}

TEST(PackedTokenPlanTest, RejectsDuplicateGenerationAndInvalidIdentity) {
  const std::array duplicate{input(1, 0, 1, 2), input(1, 1, 1, 3)};
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", resource_root(), duplicate, 2,
      PackedTokenPlanLimits{2, 2, 2}).ok());
  const std::array valid{input(1, 0, 1, 2)};
  EXPECT_FALSE(PackedTokenPlan::Create(
      0, 1, PackedTokenPhase::kDecode, "p", resource_root(), valid, 1,
      PackedTokenPlanLimits{1, 1, 1}).ok());
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 0, PackedTokenPhase::kDecode, "p", resource_root(), valid, 1,
      PackedTokenPlanLimits{1, 1, 1}).ok());
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "", resource_root(), valid, 1,
      PackedTokenPlanLimits{1, 1, 1}).ok());
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", Sha256Digest{}, valid, 1,
      PackedTokenPlanLimits{1, 1, 1}).ok());
  auto zero_input = valid;
  zero_input[0].input_digest = Sha256Digest{};
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", resource_root(), zero_input, 1,
      PackedTokenPlanLimits{1, 1, 1}).ok());
}

TEST(PackedTokenPlanTest, RejectsPhaseCountAndCapacityDrift) {
  const std::array decode_two{input(1, 0, 2, 2)};
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", resource_root(), decode_two, 2,
      PackedTokenPlanLimits{1, 2, 2}).ok());
  const std::array verify_six{input(1, 0, 6, 2)};
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kVerify, "p", resource_root(), verify_six, 6,
      PackedTokenPlanLimits{1, 6, 6}).ok());
  const std::array prefill{input(1, 0, 3, 2)};
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kPrefill, "p", resource_root(), prefill, 2,
      PackedTokenPlanLimits{1, 3, 3}).ok());
  EXPECT_FALSE(PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kPrefill, "p", resource_root(), prefill, 4,
      PackedTokenPlanLimits{1, 2, 4}).ok());
}

TEST(PackedTokenPlanTest, SemanticDigestChangesWithLogicalIdentity) {
  const std::array sequences{input(1, 0, 1, 2)};
  auto first = PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", resource_root(), sequences, 1,
      PackedTokenPlanLimits{1, 1, 1}).value();
  auto repeated = PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", resource_root(), sequences, 1,
      PackedTokenPlanLimits{1, 1, 1}).value();
  auto next = PackedTokenPlan::Create(
      1, 2, PackedTokenPhase::kDecode, "p", resource_root(), sequences, 1,
      PackedTokenPlanLimits{1, 1, 1}).value();
  EXPECT_EQ(first.semantic_digest().value(), repeated.semantic_digest().value());
  EXPECT_NE(first.semantic_digest().value(), next.semantic_digest().value());
}

}  // namespace
}  // namespace pih
