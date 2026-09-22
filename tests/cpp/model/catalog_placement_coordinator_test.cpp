#include "pih/model/catalog_placement_coordinator.h"
#include "pih/model/catalog_placement_deadline_policy.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest coordinator_root(std::byte value) {
  Sha256Digest root{};
  root.bytes.fill(value);
  return root;
}

RuntimeProfileSupervisorBootstrapManifest coordinator_bootstrap() {
  auto fds = RuntimeProfileInheritedFdManifest::Create(
      {3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14}).value();
  std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
  nonce.fill(std::byte{3});
  return RuntimeProfileSupervisorBootstrapManifest::Create(
      std::move(fds), 1, coordinator_root(std::byte{1}),
      coordinator_root(std::byte{2}), nonce).value();
}

CatalogPlacementDeadlinePolicy coordinator_deadlines() {
  return CatalogPlacementDeadlinePolicy::Create(10, 20, 30, 40, 50).value();
}

TEST(CatalogPlacementCoordinatorTest,
     AllowsOnlyOneInflightTransactionPerTargetAndRejectsReplay) {
  auto coordinator = CatalogPlacementCoordinator::Create(3).value();
  auto bootstrap = coordinator_bootstrap();
  const auto target = coordinator_root(std::byte{4});
  const auto first = coordinator_root(std::byte{5});
  const auto second = coordinator_root(std::byte{6});
  const auto deadlines = coordinator_deadlines();
  ASSERT_TRUE(coordinator.begin(first, target, bootstrap, deadlines, 100).ok());
  auto recovery = coordinator.recovery_snapshot(first);
  ASSERT_TRUE(recovery.ok());
  EXPECT_EQ(recovery->transaction_id, first);
  EXPECT_EQ(recovery->absolute_deadlines[4], 150U);
  EXPECT_FALSE(coordinator.begin(second, target, bootstrap, deadlines, 100).ok());
  EXPECT_FALSE(coordinator.begin(first, coordinator_root(std::byte{7}),
                                 bootstrap, deadlines, 100).ok());
  ASSERT_TRUE(coordinator.abort(first).ok());
  EXPECT_EQ(*coordinator.state(first), CatalogPlacementState::kQuarantined);
  EXPECT_FALSE(coordinator.begin(second, target, bootstrap, deadlines, 100).ok());
  EXPECT_FALSE(coordinator.state(coordinator_root(std::byte{8})).ok());
}

TEST(CatalogPlacementCoordinatorTest, BoundsRetainedReplayHistory) {
  auto coordinator = CatalogPlacementCoordinator::Create(1).value();
  auto bootstrap = coordinator_bootstrap();
  const auto deadlines = coordinator_deadlines();
  const auto first = coordinator_root(std::byte{5});
  ASSERT_TRUE(coordinator.begin(first, coordinator_root(std::byte{4}),
                                bootstrap, deadlines, 100).ok());
  ASSERT_TRUE(coordinator.fail(first).ok());
  EXPECT_EQ(*coordinator.state(first), CatalogPlacementState::kQuarantined);
  EXPECT_FALSE(coordinator.begin(coordinator_root(std::byte{6}),
                                 coordinator_root(std::byte{7}), bootstrap,
                                 deadlines, 100).ok());
}

TEST(CatalogPlacementCoordinatorTest,
     DeadlinePolicyFreezesFiniteAbsoluteMonotonicDeadlines) {
  auto policy = CatalogPlacementDeadlinePolicy::Create(10, 20, 30, 40, 50);
  ASSERT_TRUE(policy.ok());
  EXPECT_EQ(*policy->absolute_deadlines(100),
            (std::array<std::uint64_t, 5>{110, 120, 130, 140, 150}));
  EXPECT_FALSE(CatalogPlacementDeadlinePolicy::Create(10, 9, 30, 40, 50).ok());
  EXPECT_FALSE(CatalogPlacementDeadlinePolicy::Create(10, 20, 30, 51, 50).ok());
  EXPECT_FALSE(CatalogPlacementDeadlinePolicy::Create(
      10, 20, 30, 40, CatalogPlacementDeadlinePolicy::kMaximumTotalMs + 1).ok());
}

TEST(CatalogPlacementCoordinatorTest, ExpiredDeadlineEntersAbortingWithoutRelease) {
  auto coordinator = CatalogPlacementCoordinator::Create(2).value();
  const auto transaction = coordinator_root(std::byte{5});
  const auto target = coordinator_root(std::byte{4});
  const auto deadlines = coordinator_deadlines();
  ASSERT_TRUE(coordinator.begin(transaction, target, coordinator_bootstrap(),
                                deadlines, 100).ok());
  EXPECT_FALSE(coordinator.enforce_deadline(transaction, 110).ok());
  EXPECT_EQ(*coordinator.state(transaction), CatalogPlacementState::kAborting);
  EXPECT_FALSE(coordinator.fail(transaction).ok());
  EXPECT_EQ(*coordinator.state(transaction), CatalogPlacementState::kAborting);
  ASSERT_TRUE(coordinator.quarantine(transaction).ok());
  EXPECT_EQ(*coordinator.state(transaction), CatalogPlacementState::kQuarantined);
  EXPECT_FALSE(coordinator.begin(coordinator_root(std::byte{6}), target,
                                coordinator_bootstrap(), deadlines, 100).ok());
}

TEST(CatalogPlacementCoordinatorTest, RegressedDeadlineClockEntersAborting) {
  auto coordinator = CatalogPlacementCoordinator::Create(1).value();
  const auto transaction = coordinator_root(std::byte{5});
  ASSERT_TRUE(coordinator.begin(transaction, coordinator_root(std::byte{4}),
                                coordinator_bootstrap(), coordinator_deadlines(),
                                100).ok());
  EXPECT_FALSE(coordinator.enforce_deadline(transaction, 99).ok());
  EXPECT_EQ(*coordinator.state(transaction), CatalogPlacementState::kAborting);
}

}  // namespace
}  // namespace pih
