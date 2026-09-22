#include "pih/model/deepseek_control_plane.h"

#include <gtest/gtest.h>

namespace pih { namespace {

DeepSeekControlPlane make_control(std::uint32_t world_size = 2) {
  auto capacity = DeepSeekPipelineCapacity::Create(world_size, 8, 8, 1, false);
  return DeepSeekControlPlane::Create(7, std::move(*capacity)).value();
}

TEST(DeepSeekControlPlaneTest, OwnsOnePlanThroughCompletion) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.submit(11, 1).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}}).ok());
  EXPECT_TRUE(control.execution_live());
  EXPECT_FALSE(control.prepare_plan(
      {7, 2, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.stage_ready(1).ok());
  EXPECT_TRUE(control.validate_commit_plan().ok());
  EXPECT_EQ(control.execution_state().value(),
            DeepSeekPipelineCoordinatorState::kReadyToCommit);
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.cancel(10, 1).ok());
  ASSERT_TRUE(control.stage_complete(1).ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  EXPECT_FALSE(control.execution_live());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kCancelled);
  EXPECT_EQ(control.request_state(11, 1).value(),
            DeepSeekRequestState::kCompleted);
}

TEST(DeepSeekControlPlaneTest, PreCommitCancelAbortsExecutionOwner) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.submit(11, 1).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 2, 2}, {{10, 1}, {11, 1}}).ok());
  ASSERT_TRUE(control.cancel(10, 1).ok());
  EXPECT_FALSE(control.execution_live());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kCancelled);
  EXPECT_EQ(control.request_state(11, 1).value(),
            DeepSeekRequestState::kAdmitted);
}

TEST(DeepSeekControlPlaneTest, PostCommitFailureRetainsPoisonedOwner) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.submit(11, 1).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  EXPECT_FALSE(control.stage_failed(
      0, Status::Internal("injected worker failure")).ok());
  EXPECT_TRUE(control.execution_live());
  EXPECT_EQ(control.execution_state().value(),
            DeepSeekPipelineCoordinatorState::kPoisoned);
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kDraining);
  EXPECT_EQ(control.request_state(11, 1).value(),
            DeepSeekRequestState::kFailed);
  EXPECT_TRUE(control.acknowledge_output_plan(1).ok());
  EXPECT_FALSE(control.acknowledge_output_plan(1).ok());
}

TEST(DeepSeekControlPlaneTest, RejectsEpochMismatchBeforeReservation) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  EXPECT_FALSE(control.prepare_plan(
      {8, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1}}).ok());
  EXPECT_FALSE(control.execution_live());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kAdmitted);
}

TEST(DeepSeekControlPlaneTest, UnrelatedCancelDoesNotAbortPreparedPlan) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.submit(11, 1).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1}}).ok());
  ASSERT_TRUE(control.cancel(11, 1).ok());
  EXPECT_TRUE(control.execution_live());
  EXPECT_EQ(control.execution_state().value(),
            DeepSeekPipelineCoordinatorState::kPreparing);
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kPrepared);
  EXPECT_EQ(control.request_state(11, 1).value(),
            DeepSeekRequestState::kCancelled);
}

TEST(DeepSeekControlPlaneTest, PublishesTerminalAfterNonTerminalPlan) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kAdmitted);
  ASSERT_TRUE(control.finish_request(10, 1).ok());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kCompleted);
  EXPECT_FALSE(control.validate_retire(10, 1).ok());
  EXPECT_EQ(control.active_request_count(), 1U);
  EXPECT_FALSE(control.retire(10, 1).ok());
  ASSERT_TRUE(control.acknowledge_output_plan(1).ok());
  EXPECT_TRUE(control.validate_retire(10, 1).ok());
  EXPECT_TRUE(control.retire(10, 1).ok());
}

TEST(DeepSeekControlPlaneTest, CommitsAcceptedTokensOnlyAfterAllRanksComplete) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.stage_ready(1).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.stage_accepted_tokens(
      1, {{{42}, DeepSeekFinishReason::kNone}}).ok());
  ASSERT_TRUE(control.prepare_stage_complete(0).ok());
  EXPECT_EQ(control.execution_state().value(),
            DeepSeekPipelineCoordinatorState::kCommitted);
  EXPECT_TRUE(control.accepted_token_snapshot(10, 1)->token_ids.empty());
  ASSERT_TRUE(control.stage_complete(0).ok());
  auto before = control.accepted_token_snapshot(10, 1);
  ASSERT_TRUE(before.ok());
  EXPECT_TRUE(before->token_ids.empty());
  ASSERT_TRUE(control.stage_complete(1).ok());
  auto after = control.accepted_token_snapshot(10, 1);
  ASSERT_TRUE(after.ok());
  EXPECT_EQ(after->token_ids, std::vector<std::uint32_t>({42}));
  EXPECT_EQ(after->accepted_completion_count, 1U);
  EXPECT_EQ(after->model_processed_length, 2U);
  EXPECT_EQ(after->pending_input_token, 42U);
  EXPECT_EQ(after->finish_reason, DeepSeekFinishReason::kNone);
  EXPECT_EQ(after->minimum_completion_tokens, 0U);
}

TEST(DeepSeekControlPlaneTest,
     AdvancesStochasticOrdinalOnlyWithAcceptedTokenCommit) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
  ASSERT_TRUE(control.configure_sampling(
      10, 1, {9, DeepSeekSamplingMode::kStochastic, 123, 1.0F}).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.stage_ready(1).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.stage_accepted_tokens(
      1, {{{42}, DeepSeekFinishReason::kNone,
           {{9, 0, -0.25F, 0x12345678U}}}}).ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  EXPECT_EQ(control.accepted_token_snapshot(10, 1)->sample_ordinal, 0U);
  ASSERT_TRUE(control.stage_complete(1).ok());
  EXPECT_EQ(control.accepted_token_snapshot(10, 1)->sample_ordinal, 1U);
}

TEST(DeepSeekControlPlaneTest, RejectsInvalidGreedyTopLogprobContract) {
  auto without_logprobs = make_control();
  ASSERT_TRUE(without_logprobs.submit(10, 1).ok());
  ASSERT_TRUE(without_logprobs.configure_ledger(10, 1, 2, 3, 0).ok());
  EXPECT_FALSE(without_logprobs.configure_sampling(
      10, 1,
      {9, DeepSeekSamplingMode::kGreedy, 0, 0.0F, 1.0F,
       std::nullopt, false, 1}).ok());

  auto oversized = make_control();
  ASSERT_TRUE(oversized.submit(10, 1).ok());
  ASSERT_TRUE(oversized.configure_ledger(10, 1, 2, 3, 0).ok());
  EXPECT_FALSE(oversized.configure_sampling(
      10, 1,
      {9, DeepSeekSamplingMode::kGreedy, 0, 0.0F, 1.0F,
       std::nullopt, true, 21}).ok());
}

TEST(DeepSeekControlPlaneTest, SealsSamplingConfigurationBeforePrepare) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1}}).ok());
  EXPECT_FALSE(control.configure_sampling(
      10, 1, {9, DeepSeekSamplingMode::kGreedy, 0, 0.0F}).ok());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kPrepared);
  EXPECT_FALSE(control.stage_reject(
      0, Status::Unavailable("test rollback")).ok());
  EXPECT_FALSE(control.execution_live());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kAdmitted);
}

TEST(DeepSeekControlPlaneTest, RetainsCanonicalRequestStopTokens) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
  DeepSeekRequestSamplingConfig config{
      9, DeepSeekSamplingMode::kGreedy, 0, 0.0F};
  config.stop_token_ids[0] = 17;
  config.stop_token_ids[1] = 23;
  config.stop_token_count = 2;
  ASSERT_TRUE(control.configure_sampling(10, 1, config).ok());
  const auto retained = control.sampling_config(10, 1);
  ASSERT_TRUE(retained.ok());
  EXPECT_EQ(retained->stop_token_count, 2U);
  EXPECT_EQ(retained->stop_token_ids[0], 17U);
  EXPECT_EQ(retained->stop_token_ids[1], 23U);
}

TEST(DeepSeekControlPlaneTest, RejectsNonCanonicalRequestStopTokens) {
  auto rejected = [](DeepSeekRequestSamplingConfig config) {
    auto control = make_control();
    EXPECT_TRUE(control.submit(10, 1).ok());
    EXPECT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
    return control.configure_sampling(10, 1, config);
  };

  DeepSeekRequestSamplingConfig duplicate{
      9, DeepSeekSamplingMode::kGreedy, 0, 0.0F};
  duplicate.stop_token_ids[0] = 17;
  duplicate.stop_token_ids[1] = 17;
  duplicate.stop_token_count = 2;
  EXPECT_FALSE(rejected(duplicate).ok());

  DeepSeekRequestSamplingConfig outside_vocabulary{
      9, DeepSeekSamplingMode::kGreedy, 0, 0.0F};
  outside_vocabulary.stop_token_ids[0] = 129280;
  outside_vocabulary.stop_token_count = 1;
  EXPECT_FALSE(rejected(outside_vocabulary).ok());

  DeepSeekRequestSamplingConfig nonzero_tail{
      9, DeepSeekSamplingMode::kGreedy, 0, 0.0F};
  nonzero_tail.stop_token_ids[1] = 23;
  nonzero_tail.stop_token_count = 1;
  EXPECT_FALSE(rejected(nonzero_tail).ok());
}

TEST(DeepSeekControlPlaneTest,
     CommitsGreedyLogprobAtomicallyWithoutAdvancingOrdinal) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
  ASSERT_TRUE(control.configure_sampling(
      10, 1, {9, DeepSeekSamplingMode::kGreedy, 0, 0.0F}).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.stage_ready(1).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  DeepSeekAcceptedTokenBatch batch{{42}, DeepSeekFinishReason::kNone};
  batch.selected_logprobs.push_back({9, -0.25F});
  ASSERT_TRUE(control.stage_accepted_tokens(1, {std::move(batch)}).ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  EXPECT_TRUE(control.accepted_token_snapshot(10, 1)->selected_logprobs.empty());
  ASSERT_TRUE(control.stage_complete(1).ok());
  const auto snapshot = control.accepted_token_snapshot(10, 1);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_EQ(snapshot->selected_logprobs, std::vector<float>({-0.25F}));
  EXPECT_EQ(snapshot->sample_ordinal, 0U);
}

TEST(DeepSeekControlPlaneTest, CommitsRankedTopLogprobsWithSelectedToken) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 2, 3, 0).ok());
  ASSERT_TRUE(control.configure_sampling(
      10, 1,
      {9, DeepSeekSamplingMode::kStochastic, 123, 1.0F, 1.0F,
       std::nullopt, true, 2}).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.stage_ready(1).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  DeepSeekAcceptedTokenBatch batch{{42}, DeepSeekFinishReason::kNone};
  batch.selected_logprobs.push_back(
      {9, -0.25F, {{42, -0.25F, 1}, {7, -1.25F, 2}}});
  batch.sampling_receipts.push_back({9, 0, -0.25F, 0x12345678U});
  ASSERT_TRUE(control.stage_accepted_tokens(1, {std::move(batch)}).ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  ASSERT_TRUE(control.stage_complete(1).ok());
  const auto snapshot = control.accepted_token_snapshot(10, 1);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_EQ(snapshot->top_logprobs.size(), 1U);
  ASSERT_EQ(snapshot->top_logprobs[0].size(), 2U);
  EXPECT_EQ(snapshot->top_logprobs[0][0].token_id, 42U);
  EXPECT_FLOAT_EQ(snapshot->top_logprobs[0][0].logprob, -0.25F);
  EXPECT_EQ(snapshot->top_logprobs[0][0].rank, 1U);
  EXPECT_EQ(snapshot->top_logprobs[0][1].token_id, 7U);
}

TEST(DeepSeekControlPlaneTest, DiscardsCancelledTentativeTokens) {
  auto control = make_control();
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 1, 2, 0).ok());
  ASSERT_TRUE(control.configure_sampling(
      10, 1, {9, DeepSeekSamplingMode::kStochastic, 123, 1.0F}).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.stage_ready(1).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.stage_accepted_tokens(
      1, {{{99}, DeepSeekFinishReason::kNone,
           {{9, 0, -0.5F, 0x87654321U}}}}).ok());
  ASSERT_TRUE(control.cancel(10, 1).ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  ASSERT_TRUE(control.stage_complete(1).ok());
  auto snapshot = control.accepted_token_snapshot(10, 1);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_TRUE(snapshot->token_ids.empty());
  EXPECT_EQ(snapshot->sample_ordinal, 0U);
}

TEST(DeepSeekControlPlaneTest, StopWinsOverLengthAtSameAcceptedToken) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 1, 1, 0).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.stage_accepted_tokens(
      1, {{{129279}, DeepSeekFinishReason::kStop}}).ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  auto snapshot = control.accepted_token_snapshot(10, 1);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_EQ(snapshot->finish_reason, DeepSeekFinishReason::kStop);
  EXPECT_FALSE(snapshot->pending_input_token.has_value());
  EXPECT_EQ(snapshot->model_processed_length, 1U);
}

TEST(DeepSeekControlPlaneTest, RejectsQwenOnlyVocabularyTail) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.configure_ledger(10, 1, 1, 1, 0).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kDecode, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  EXPECT_FALSE(control.stage_accepted_tokens(
      1, {{{129280}, DeepSeekFinishReason::kLength}}).ok());
  EXPECT_TRUE(control.accepted_token_snapshot(10, 1)->token_ids.empty());
}

TEST(DeepSeekControlPlaneTest, OutputBurstRequiresExactPostTransferAck) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  ASSERT_TRUE(control.prepare_plan(
      {7, 1, DeepSeekPlanPhase::kPrefill, 1, 1}, {{10, 1, false}}).ok());
  ASSERT_TRUE(control.stage_ready(0).ok());
  ASSERT_TRUE(control.commit_plan().ok());
  ASSERT_TRUE(control.stage_complete(0).ok());
  EXPECT_FALSE(control.acknowledge_output_plan(2).ok());
  ASSERT_TRUE(control.acknowledge_output_plan(1).ok());
  EXPECT_FALSE(control.acknowledge_output_plan(1).ok());
}

TEST(DeepSeekControlPlaneTest, TwoUnackedBurstsBackpressureThirdPrepare) {
  auto control = make_control(1);
  ASSERT_TRUE(control.submit(10, 1).ok());
  for (std::uint64_t plan = 1; plan <= 2; ++plan) {
    ASSERT_TRUE(control.prepare_plan(
        {7, plan, DeepSeekPlanPhase::kPrefill, 1, 1},
        {{10, 1, false}}).ok());
    ASSERT_TRUE(control.stage_ready(0).ok());
    ASSERT_TRUE(control.commit_plan().ok());
    ASSERT_TRUE(control.stage_complete(0).ok());
  }
  EXPECT_FALSE(control.prepare_plan(
      {7, 3, DeepSeekPlanPhase::kPrefill, 1, 1}, {{10, 1, false}}).ok());
  EXPECT_EQ(control.request_state(10, 1).value(),
            DeepSeekRequestState::kAdmitted);
  ASSERT_TRUE(control.acknowledge_output_plan(1).ok());
  EXPECT_TRUE(control.prepare_plan(
      {7, 3, DeepSeekPlanPhase::kPrefill, 1, 1}, {{10, 1, false}}).ok());
}

} }  // namespace pih
