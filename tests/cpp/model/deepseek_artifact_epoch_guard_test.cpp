#include "pih/model/deepseek_artifact_epoch_guard.h"
#include <gtest/gtest.h>
#include <stdexcept>

namespace pih { namespace {
class CapabilityPoller final : public DeepSeekEngineArtifactPoller {
 public:
  std::uint32_t world_size() const noexcept override { return world; }
  Status poll() override {
    ++calls;
    if (throws) throw std::runtime_error("provider failure");
    if (invalidate) invalidate->report_worker_lost(0);
    return result;
  }
  std::uint32_t world = 1, calls = 0;
  bool throws = false;
  Status result = Status::Ok();
  DeepSeekArtifactEpochGuard* invalidate = nullptr;
};

TEST(DeepSeekArtifactEpochGuardTest, LeaseFailureIrreversiblyFailsEpoch) {
  CapabilityPoller poller;
  auto guard = DeepSeekArtifactEpochGuard::Create(7, 100, 1).value();
  EXPECT_FALSE(guard.poll(poller).ok());
  EXPECT_EQ(poller.calls, 0U);
  ASSERT_TRUE(guard.mark_ready().ok());
  EXPECT_TRUE(guard.poll(poller).ok());
  poller.result = Status::Unavailable("lease changed");
  EXPECT_FALSE(guard.poll(poller).ok());
  EXPECT_EQ(guard.state(), ArtifactLeaseEpochState::kFailed);
  EXPECT_EQ(guard.failure(), ArtifactLeaseEpochFailure::kLeaseIntegrity);
  EXPECT_FALSE(guard.admission_allowed());
  EXPECT_EQ(guard.public_error(), "engine_artifact_integrity_failed");
  EXPECT_FALSE(guard.mark_ready().ok());
  poller.result = Status::Ok();
  EXPECT_FALSE(guard.poll(poller).ok());
  EXPECT_EQ(poller.calls, 2U);
}
TEST(DeepSeekArtifactEpochGuardTest, WrongTopologyFailsBeforeProviderCall) {
  CapabilityPoller poller;
  auto guard = DeepSeekArtifactEpochGuard::Create(9, 100, 2).value();
  ASSERT_TRUE(guard.mark_ready().ok());
  EXPECT_FALSE(guard.poll(poller).ok());
  EXPECT_EQ(poller.calls, 0U);
  EXPECT_EQ(guard.failure(), ArtifactLeaseEpochFailure::kLeaseIntegrity);
}
TEST(DeepSeekArtifactEpochGuardTest, ProviderExceptionFailsClosed) {
  CapabilityPoller poller; poller.throws = true;
  auto guard = DeepSeekArtifactEpochGuard::Create(9, 100, 1).value();
  ASSERT_TRUE(guard.mark_ready().ok());
  EXPECT_FALSE(guard.poll(poller).ok());
  EXPECT_EQ(guard.failure(), ArtifactLeaseEpochFailure::kLeaseIntegrity);
}
TEST(DeepSeekArtifactEpochGuardTest, ConcurrentReporterPreventsSuccessfulAdmission) {
  CapabilityPoller poller;
  auto guard = DeepSeekArtifactEpochGuard::Create(9, 100, 1).value();
  poller.invalidate = &guard;
  ASSERT_TRUE(guard.mark_ready().ok());
  EXPECT_FALSE(guard.poll(poller).ok());
  EXPECT_EQ(guard.failure(), ArtifactLeaseEpochFailure::kWorkerLost);
}
TEST(DeepSeekArtifactEpochGuardTest, MappingFaultFailsAllRanks) {
  auto guard = DeepSeekArtifactEpochGuard::Create(8, 250, 4).value();
  ASSERT_TRUE(guard.mark_ready().ok());
  EXPECT_TRUE(guard.report_mapping_fault(2).ok());
  EXPECT_EQ(guard.failure(), ArtifactLeaseEpochFailure::kMappingFault);
  EXPECT_FALSE(guard.report_mapping_fault(4).ok());
  EXPECT_FALSE(guard.admission_allowed());
  EXPECT_FALSE(guard.report_lease_integrity_failure().ok());
  EXPECT_EQ(guard.failure(), ArtifactLeaseEpochFailure::kMappingFault);
}
TEST(DeepSeekArtifactEpochGuardTest, RejectsInvalidEpochConfiguration) {
  EXPECT_FALSE(DeepSeekArtifactEpochGuard::Create(0, 100, 1).ok());
  EXPECT_FALSE(DeepSeekArtifactEpochGuard::Create(1, 0, 1).ok());
  EXPECT_FALSE(DeepSeekArtifactEpochGuard::Create(1, 100, 0).ok());
  EXPECT_FALSE(DeepSeekArtifactEpochGuard::Create(1, 100, 5).ok());
}
} }  // namespace pih
