#include "pih/model/deepseek_rank_plan_reservation.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih { namespace {

class TrackingLease final : public DeepSeekRankPlanResourceLease {
 public:
  TrackingLease(DeepSeekRankPlanResourceKind kind,
                std::vector<DeepSeekRankPlanResourceKind>& released)
      : kind_(kind), released_(&released) {}
  ~TrackingLease() override { released_->push_back(kind_); }
 private:
  DeepSeekRankPlanResourceKind kind_;
  std::vector<DeepSeekRankPlanResourceKind>* released_;
};

class TrackingProvider final : public DeepSeekRankPlanResourceProvider {
 public:
  Result<std::unique_ptr<DeepSeekRankPlanResourceLease>> reserve(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      const DeepSeekStagePlan& stage,
      DeepSeekRankPlanResourceRequirement requirement) override {
    if (descriptor.plan_sequence != 9 || stage.rank != expected_rank) {
      return Status::FailedPrecondition("plan identity drifted");
    }
    requested.push_back(requirement);
    if (requested.size() == fail_at) {
      return Status::ResourceExhausted("injected rank reservation failure");
    }
    return std::unique_ptr<DeepSeekRankPlanResourceLease>(
        new TrackingLease(requirement.kind, released));
  }
  std::uint32_t expected_rank = 0;
  std::size_t fail_at = 99;
  std::vector<DeepSeekRankPlanResourceRequirement> requested;
  std::vector<DeepSeekRankPlanResourceKind> released;
};

TEST(DeepSeekRankPlanReservationTest,
     AtomicallyOwnsCompleteHostSpillResourceVector) {
  auto pipeline = DeepSeekPipelinePlan::Create(2, false).value();
  TrackingProvider provider;
  provider.expected_rank = 0;
  auto reservation = DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kVerify, 5, 3}, pipeline.rank(0), 2,
      DeepSeekRoutedExpertResidency::kHostSpill, 4096, 8192, 16384,
      provider);
  ASSERT_TRUE(reservation.ok()) << reservation.status().message();
  EXPECT_EQ(provider.requested.size(), 8U);
  EXPECT_EQ(provider.requested.front().kind,
            DeepSeekRankPlanResourceKind::kOutgoingActivation);
  EXPECT_EQ(provider.requested.back().kind,
            DeepSeekRankPlanResourceKind::kPinnedStagingCredit);
  EXPECT_TRUE(reservation->validate_commit().ok());
  EXPECT_EQ(reservation->state(),
            DeepSeekRankPlanReservationState::kPrepared);
  EXPECT_TRUE(reservation->commit().ok());
  EXPECT_FALSE(reservation->validate_commit().ok());
  reservation->release();
  EXPECT_EQ(provider.released.size(), provider.requested.size());
  EXPECT_EQ(provider.released.front(), provider.requested.back().kind);
  EXPECT_EQ(reservation->state(),
            DeepSeekRankPlanReservationState::kReleased);
}

TEST(DeepSeekRankPlanReservationTest, RollsBackEveryPriorLeaseOnFailure) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  TrackingProvider provider;
  provider.fail_at = 4;
  auto reservation = DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, pipeline.rank(0), 1,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 8192, 16384,
      provider);
  ASSERT_FALSE(reservation.ok());
  EXPECT_EQ(provider.requested.size(), 4U);
  EXPECT_EQ(provider.released.size(), 3U);
  EXPECT_EQ(provider.released[0], provider.requested[2].kind);
  EXPECT_EQ(provider.released[2], provider.requested[0].kind);
}

TEST(DeepSeekRankPlanReservationTest, RejectsIncompletePhaseCapacity) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  TrackingProvider provider;
  EXPECT_FALSE(DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kVerify, 5, 1}, pipeline.rank(0), 1,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 8192, 16384,
      provider).ok());
  EXPECT_FALSE(DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, pipeline.rank(0), 1,
      DeepSeekRoutedExpertResidency::kHostSpill, 4096, 8192, 16384,
      provider).ok());
  EXPECT_TRUE(provider.requested.empty());
}

TEST(DeepSeekRankPlanReservationTest,
     MoveAssignmentReleasesOldVectorBeforeTakingOwnership) {
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  TrackingProvider first_provider;
  TrackingProvider second_provider;
  auto first = DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, pipeline.rank(0), 1,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 8192, 16384,
      first_provider);
  auto second = DeepSeekRankPlanReservation::Prepare(
      {7, 9, DeepSeekPlanPhase::kDecode, 1, 1}, pipeline.rank(0), 1,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 8192, 16384,
      second_provider);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  const auto first_count = first_provider.requested.size();
  const auto second_count = second_provider.requested.size();

  *first = std::move(*second);
  EXPECT_EQ(first_provider.released.size(), first_count);
  EXPECT_TRUE(second_provider.released.empty());
  EXPECT_EQ(second->state(), DeepSeekRankPlanReservationState::kReleased);
  first->release();
  EXPECT_EQ(second_provider.released.size(), second_count);
}

} }  // namespace pih
