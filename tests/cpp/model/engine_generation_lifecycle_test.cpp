#include "pih/model/engine_generation_lifecycle.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

EngineRestartPolicy policy() { return {10, 40, 100, 3}; }

Sha256Digest lease() {
  Sha256Digest value{}; value.bytes.fill(std::byte{9}); return value;
}

EngineGenerationQuiescenceObservation verified() {
  return {true, true, true, true, true, true, true, true, true, true};
}

EngineGenerationQuiescenceReceipt receipt(
    EngineGenerationQuiescenceObservation observation, std::uint64_t now,
    std::uint64_t generation = 1, std::uint64_t sample = 1) {
  return {generation, lease(), sample, now - 1, now, observation};
}

void reach_dead(EngineGenerationLifecycle& value) {
  ASSERT_TRUE(value.begin_start().ok());
  ASSERT_TRUE(value.publish_ready().ok());
  ASSERT_TRUE(value.fail_generation(true).ok());
  ASSERT_TRUE(value.controller_reaped().ok());
  ASSERT_TRUE(value.begin_quiescence_check().ok());
}

TEST(EngineGenerationLifecycleTest, RequiresCompleteQuiescenceBeforeBackoff) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  reach_dead(lifecycle);
  auto pending = verified(); pending.network_baseline = false;
  auto result = lifecycle.observe_quiescence(receipt(pending, 100), 100);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, EngineQuiescenceState::kPending);
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kQuiescenceCheck);
  result = lifecycle.observe_quiescence(receipt(verified(), 101, 1, 2), 101);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, EngineQuiescenceState::kVerified);
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kBackoff);
  EXPECT_EQ(lifecycle.retry_not_before_ns(), 111U);
  EXPECT_EQ(lifecycle.finish_backoff(110, 2).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(lifecycle.finish_backoff(111, 2).ok());
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kLeased);
  EXPECT_EQ(lifecycle.generation(), 2U);
}

TEST(EngineGenerationLifecycleTest, UnknownVisibilityQuarantines) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  reach_dead(lifecycle);
  auto unknown = verified(); unknown.visibility_complete = false;
  auto result = lifecycle.observe_quiescence(receipt(unknown, 100), 100);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, EngineQuiescenceState::kUnknown);
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kQuarantined);
}

TEST(EngineGenerationLifecycleTest, AppliesFiniteBackoffAndBurstQuarantine) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  for (std::uint64_t attempt = 0; attempt < 3; ++attempt) {
    reach_dead(lifecycle);
    const auto now = 100 + attempt * 20;
    ASSERT_EQ(*lifecycle.observe_quiescence(
                  receipt(verified(), now, attempt + 1, attempt + 1), now),
              EngineQuiescenceState::kVerified);
    EXPECT_EQ(lifecycle.retry_not_before_ns(),
              now + (attempt == 0 ? 10 : attempt == 1 ? 20 : 40));
    ASSERT_TRUE(lifecycle.finish_backoff(lifecycle.retry_not_before_ns(),
                                         attempt + 2).ok());
  }
  reach_dead(lifecycle);
  ASSERT_EQ(*lifecycle.observe_quiescence(receipt(verified(), 160, 4, 4), 160),
            EngineQuiescenceState::kVerified);
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kQuarantined);
}

TEST(EngineGenerationLifecycleTest, RejectsInvalidPolicyAndStateSkipping) {
  EXPECT_FALSE(EngineGenerationLifecycle::Create(0, lease(), policy()).ok());
  EXPECT_FALSE(EngineGenerationLifecycle::Create(1, {}, policy()).ok());
  EXPECT_FALSE(EngineGenerationLifecycle::Create(1, lease(), {0, 40, 100, 3}).ok());
  EXPECT_FALSE(EngineGenerationLifecycle::Create(1, lease(), {50, 40, 100, 3}).ok());
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  EXPECT_FALSE(lifecycle.publish_ready().ok());
  EXPECT_FALSE(lifecycle.controller_reaped().ok());
}

TEST(EngineGenerationLifecycleTest, PlannedStopDoesNotAutoRestart) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  ASSERT_TRUE(lifecycle.begin_start().ok());
  ASSERT_TRUE(lifecycle.publish_ready().ok());
  ASSERT_TRUE(lifecycle.begin_drain().ok());
  ASSERT_TRUE(lifecycle.begin_stop().ok());
  ASSERT_TRUE(lifecycle.controller_reaped().ok());
  ASSERT_TRUE(lifecycle.begin_quiescence_check().ok());
  ASSERT_EQ(*lifecycle.observe_quiescence(receipt(verified(), 100), 100),
            EngineQuiescenceState::kVerified);
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kDead);
  EXPECT_EQ(lifecycle.retry_not_before_ns(), 0U);
}

TEST(EngineGenerationLifecycleTest, NonretryableFailureDoesNotAutoRestart) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  ASSERT_TRUE(lifecycle.begin_start().ok());
  ASSERT_TRUE(lifecycle.fail_generation(false).ok());
  ASSERT_TRUE(lifecycle.controller_reaped().ok());
  ASSERT_TRUE(lifecycle.begin_quiescence_check().ok());
  ASSERT_EQ(*lifecycle.observe_quiescence(receipt(verified(), 100), 100),
            EngineQuiescenceState::kVerified);
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kDead);
}

TEST(EngineGenerationLifecycleTest, RejectsClockRegressionAndGenerationSkip) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  reach_dead(lifecycle);
  ASSERT_EQ(*lifecycle.observe_quiescence(receipt(verified(), 100), 100),
            EngineQuiescenceState::kVerified);
  EXPECT_FALSE(lifecycle.finish_backoff(110, 3).ok());
  ASSERT_TRUE(lifecycle.finish_backoff(110, 2).ok());
  reach_dead(lifecycle);
  auto result = lifecycle.observe_quiescence(receipt(verified(), 99, 2, 2), 99);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kQuarantined);
}

TEST(EngineGenerationLifecycleTest, QuarantinesMismatchedReceiptIdentity) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto lifecycle =
        EngineGenerationLifecycle::Create(1, lease(), policy()).value();
    reach_dead(lifecycle);
    auto value = receipt(verified(), 100);
    if (mutation == 0) value.generation = 2;
    if (mutation == 1) value.allocation_lease_digest.bytes[0] = std::byte{8};
    if (mutation == 2) value.sample_identity = 0;
    if (mutation == 3) value.sample_completed_ns = 101;
    auto result = lifecycle.observe_quiescence(value, 100);
    ASSERT_FALSE(result.ok()) << mutation;
    EXPECT_EQ(lifecycle.state(), EngineGenerationState::kQuarantined)
        << mutation;
  }
}

TEST(EngineGenerationLifecycleTest, RejectsReplayedOrOverlappingSample) {
  auto lifecycle = EngineGenerationLifecycle::Create(1, lease(), policy()).value();
  reach_dead(lifecycle);
  auto pending = verified(); pending.network_baseline = false;
  ASSERT_EQ(*lifecycle.observe_quiescence(receipt(pending, 100), 100),
            EngineQuiescenceState::kPending);
  auto replay = receipt(verified(), 101);
  replay.sample_identity = 1;
  EXPECT_FALSE(lifecycle.observe_quiescence(replay, 101).ok());
  EXPECT_EQ(lifecycle.state(), EngineGenerationState::kQuarantined);
}

}  // namespace
}  // namespace pih
