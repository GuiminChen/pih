#include "pih/scheduler/phase_tier_scheduler.h"

#include <array>
#include <string_view>

#include <gtest/gtest.h>

namespace pih {
namespace {
SchedulerCandidate candidate(std::uint64_t generation, PackedTokenPhase phase,
                             std::uint64_t last, std::uint64_t enqueue,
                             std::int64_t since, std::uint32_t tokens = 1,
                             SchedulerInfeasibility reason = SchedulerInfeasibility::kNone) {
  return {generation, phase, last, enqueue, since, tokens, reason};
}

PhaseTierScheduler scheduler() {
  return PhaseTierScheduler::Create({8, 4, 8, 8, 2, 100}).value();
}

Sha256Digest digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}

TEST(PhaseTierSchedulerTest, ProtectsAgedPrefillThenForcesDecode) {
  const std::array candidates{
      candidate(1, PackedTokenPhase::kPrefill, 0, 2, 0),
      candidate(2, PackedTokenPhase::kDecode, 0, 1, 0)};
  auto policy = scheduler();
  auto first = policy.select({100, 0, false, candidates}).value();
  EXPECT_EQ(first.tier, SchedulerTier::kProtectedPrefill);
  ASSERT_EQ(first.included_candidate_indices.size(), 1U);
  EXPECT_EQ(first.included_candidate_indices[0], 0U);
  auto second = policy.select({101, 0, true, candidates}).value();
  EXPECT_EQ(second.tier, SchedulerTier::kDecodeFamily);
  ASSERT_EQ(second.included_candidate_indices.size(), 1U);
  EXPECT_EQ(second.included_candidate_indices[0], 1U);
}

TEST(PhaseTierSchedulerTest, DecodeCapSelectsPrefillAndDecodeHeadFixesPhase) {
  const std::array candidates{
      candidate(3, PackedTokenPhase::kVerify, 0, 1, 90),
      candidate(2, PackedTokenPhase::kDecode, 0, 2, 90),
      candidate(1, PackedTokenPhase::kPrefill, 0, 3, 90)};
  auto policy = scheduler();
  auto capped = policy.select({99, 2, false, candidates}).value();
  EXPECT_EQ(capped.tier, SchedulerTier::kNormalPrefill);
  auto decode = policy.select({99, 1, false, candidates}).value();
  EXPECT_EQ(decode.phase, PackedTokenPhase::kVerify);
  ASSERT_EQ(decode.included_candidate_indices.size(), 1U);
  EXPECT_EQ(decode.included_candidate_indices[0], 0U);
}

TEST(PhaseTierSchedulerTest, InfeasibleProtectedDoesNotBlockDecode) {
  const std::array candidates{
      candidate(1, PackedTokenPhase::kPrefill, 0, 1, 0, 1,
                SchedulerInfeasibility::kWorkspace),
      candidate(2, PackedTokenPhase::kDecode, 0, 2, 0)};
  auto policy = scheduler();
  auto decision = policy.select({100, 0, false, candidates}).value();
  EXPECT_EQ(decision.tier, SchedulerTier::kDecodeFamily);
  EXPECT_EQ(decision.scanned_count, 2U);
  EXPECT_EQ(decision.candidate_results[0], SchedulerInfeasibility::kWorkspace);
}

TEST(PhaseTierSchedulerTest, OrdersKeysAndGreedilyHonorsTokenBound) {
  const std::array candidates{
      candidate(9, PackedTokenPhase::kPrefill, 4, 1, 90, 5),
      candidate(8, PackedTokenPhase::kPrefill, 0, 3, 90, 4),
      candidate(7, PackedTokenPhase::kPrefill, 0, 2, 90, 4)};
  auto policy = scheduler();
  auto decision = policy.select({99, 2, false, candidates}).value();
  ASSERT_EQ(decision.included_candidate_indices.size(), 2U);
  EXPECT_EQ(decision.included_candidate_indices[0], 2U);
  EXPECT_EQ(decision.included_candidate_indices[1], 1U);
  EXPECT_EQ(decision.candidate_results[0], SchedulerInfeasibility::kPlanTokenBound);
  EXPECT_EQ(decision.total_real_tokens, 8U);
}

TEST(PhaseTierSchedulerTest, RejectsInvalidSnapshotWithoutPublishing) {
  auto policy = scheduler();
  const std::array valid{candidate(1, PackedTokenPhase::kDecode, 0, 1, 0)};
  auto first = policy.select({1, 0, false, valid}).value();
  const std::array invalid{candidate(1, PackedTokenPhase::kDecode, 0, 0, 0)};
  EXPECT_FALSE(policy.select({1, 0, false, invalid}).ok());
  const std::array duplicate{
      candidate(1, PackedTokenPhase::kDecode, 0, 1, 0),
      candidate(1, PackedTokenPhase::kDecode, 0, 2, 0)};
  EXPECT_FALSE(policy.select({1, 0, false, duplicate}).ok());
  auto second = policy.select({1, 0, false, valid}).value();
  EXPECT_EQ(second.generation, first.generation + 1);
}

TEST(PhaseTierSchedulerTest, RejectsPlanSequenceCapacityAboveCandidates) {
  EXPECT_FALSE(PhaseTierScheduler::Create({1, 2, 2, 1, 1, 1}).ok());
}

TEST(PhaseTierSchedulerTest, AssemblesOnlyTheCurrentInternalDecision) {
  const std::array candidates{
      candidate(11, PackedTokenPhase::kPrefill, 0, 2, 90, 3),
      candidate(12, PackedTokenPhase::kPrefill, 0, 1, 90, 2)};
  const std::array states{
      ScheduledSequenceState{11, 7, 101, digest("a")},
      ScheduledSequenceState{12, 20, 102, digest("b")}};
  auto policy = scheduler();
  const auto decision = policy.select({99, 2, false, candidates}).value();
  auto plan = policy.assemble_current_plan(
      decision.generation,
      {4, 8, "profile-r1", digest("resources"), 8, 8}, states);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->phase(), PackedTokenPhase::kPrefill);
  EXPECT_EQ(plan->ordered_sequence_generations(),
            (std::vector<std::uint64_t>{12, 11}));
  EXPECT_EQ(plan->real_token_counts(),
            (std::vector<std::uint32_t>{2, 3}));
  EXPECT_EQ(plan->packed_offsets(),
            (std::vector<std::uint32_t>{0, 2, 5}));
}

TEST(PhaseTierSchedulerTest, RejectsStaleDecisionAndSequenceStateDrift) {
  const std::array candidates{
      candidate(11, PackedTokenPhase::kDecode, 0, 1, 0)};
  const std::array states{
      ScheduledSequenceState{11, 7, 101, digest("a")}};
  auto policy = scheduler();
  const auto old = policy.select({1, 0, false, candidates}).value();
  const auto current = policy.select({1, 0, false, candidates}).value();
  const ScheduledPlanIdentity identity{
      1, 1, "p", digest("resources"), 1, 1};
  EXPECT_FALSE(policy.assemble_current_plan(old.generation, identity, states).ok());
  auto drifted = states;
  drifted[0].sequence_generation = 12;
  EXPECT_FALSE(policy.assemble_current_plan(current.generation, identity, drifted).ok());
  EXPECT_TRUE(policy.assemble_current_plan(current.generation, identity, states).ok());
}

TEST(PhaseTierSchedulerTest, MaterializesSelectedTokensInSchedulerOrder) {
  const std::array<std::uint32_t, 3> first_tokens{4, 5, 6};
  const std::array<std::uint32_t, 2> second_tokens{8, 9};
  const std::array candidates{
      candidate(11, PackedTokenPhase::kPrefill, 0, 2, 0, 3),
      candidate(12, PackedTokenPhase::kPrefill, 0, 1, 0, 2)};
  const std::array states{
      ScheduledSequenceState{11, 7, 101,
                             packed_token_input_digest(first_tokens).value()},
      ScheduledSequenceState{12, 20, 102,
                             packed_token_input_digest(second_tokens).value()}};
  const std::array tokens{
      ScheduledSequenceTokens{11, first_tokens, false},
      ScheduledSequenceTokens{12, second_tokens, true}};
  auto policy = scheduler();
  const auto decision = policy.select({100, 2, false, candidates}).value();
  auto plan = policy.assemble_current_plan(
      decision.generation,
      {1, 1, "p", digest("resources"), 8, 8}, states).value();
  auto arena = PackedTokenMetadataArena::Create({4, 8}).value();
  auto metadata = policy.materialize_current_metadata(
      decision.generation, plan, tokens, arena);
  ASSERT_TRUE(metadata.ok()) << metadata.status().message();
  ASSERT_EQ(metadata->input_token_ids.size(), 8U);
  const std::array<std::uint32_t, 8> expected{8, 9, 4, 5, 6, 0, 0, 0};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(metadata->input_token_ids[i], expected[i]);
  }
  ASSERT_EQ(metadata->sample_row_index.size(), 1U);
  EXPECT_EQ(metadata->sample_row_index[0], 1U);
  EXPECT_EQ(metadata->query_start_offsets[1], 2U);
  EXPECT_EQ(metadata->query_start_offsets[2], 5U);
}

TEST(PhaseTierSchedulerTest, RejectsForgedPlanBeforeArenaPublication) {
  const std::array<std::uint32_t, 1> token{4};
  const std::array candidates{
      candidate(11, PackedTokenPhase::kDecode, 0, 1, 0)};
  const std::array states{ScheduledSequenceState{
      11, 7, 101, packed_token_input_digest(token).value()}};
  const std::array tokens{ScheduledSequenceTokens{11, token, true}};
  auto policy = scheduler();
  const auto decision = policy.select({1, 0, false, candidates}).value();
  auto plan = policy.assemble_current_plan(
      decision.generation,
      {1, 1, "p", digest("resources"), 1, 1}, states).value();
  auto forged_inputs = std::array{PackedSequenceInput{
      99, 7, 1, 101, packed_token_input_digest(token).value()}};
  auto forged = PackedTokenPlan::Create(
      1, 1, PackedTokenPhase::kDecode, "p", digest("resources"),
      forged_inputs, 1, {1, 1, 1}).value();
  auto arena = PackedTokenMetadataArena::Create({1, 1}).value();
  EXPECT_FALSE(policy.materialize_current_metadata(
      decision.generation, forged, tokens, arena).ok());
  EXPECT_TRUE(policy.materialize_current_metadata(
      decision.generation, plan, tokens, arena).ok());
}
}  // namespace
}  // namespace pih
