#include "pih/model/engine_cuda_owner_ledger_registry.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest registry_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

class RegistryClock final : public EngineCudaOwnerLedgerClock {
 public:
  Result<std::uint64_t> monotonic_ns() override { return values.at(next++); }
  std::array<std::uint64_t, 4> values{100, 110, 120, 130};
  std::size_t next = 0;
};

TEST(EngineCudaOwnerLedgerRegistryTest,
     CapturesRegisteredAndReleasedOwnersInOneSnapshot) {
  RegistryClock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  ASSERT_TRUE(registry.register_pinned(60, registry_digest(9), 0, 4096).ok());
  ASSERT_TRUE(registry.register_allocation(70, registry_digest(9), 8192).ok());
  ASSERT_TRUE(registry.release_pinned(60).ok());
  const std::array<std::uint64_t, 1> pinned{60};
  const std::array<std::uint64_t, 1> allocations{70};

  auto snapshot = registry.capture(pinned, allocations);

  ASSERT_TRUE(snapshot.ok());
  EXPECT_EQ(snapshot->sample_identity, 1U);
  EXPECT_EQ(snapshot->sample_started_ns, 100U);
  EXPECT_EQ(snapshot->sample_completed_ns, 110U);
  ASSERT_EQ(snapshot->pinned.size(), 1U);
  EXPECT_FALSE(snapshot->pinned[0].registered);
  EXPECT_TRUE(snapshot->pinned[0].owner_counter_visible);
  EXPECT_EQ(snapshot->pinned[0].registered_bytes, 0U);
  ASSERT_EQ(snapshot->allocations.size(), 1U);
  EXPECT_TRUE(snapshot->allocations[0].allocated);
  EXPECT_EQ(snapshot->allocations[0].allocated_bytes, 8192U);
}

TEST(EngineCudaOwnerLedgerRegistryTest,
     RejectsDuplicateLiveRegistrationAndUnknownRelease) {
  RegistryClock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  ASSERT_TRUE(registry.register_allocation(70, registry_digest(9), 8192).ok());
  EXPECT_EQ(registry.register_allocation(70, registry_digest(9), 8192)
                .code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(registry.release_allocation(71).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(registry.release_allocation(70).ok());
  EXPECT_EQ(registry.release_allocation(70).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(registry.register_allocation(70, registry_digest(9), 8192).code(),
            StatusCode::kFailedPrecondition);
}

TEST(EngineCudaOwnerLedgerRegistryTest,
     RejectsCrossCategoryIdentityReuseAndInvalidMetadata) {
  RegistryClock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  EXPECT_EQ(registry.register_pinned(0, registry_digest(9), 0, 4096).code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(registry.register_pinned(60, {}, 0, 4096).code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(registry.register_pinned(60, registry_digest(9), -1, 4096).code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(registry.register_allocation(70, registry_digest(9), 0).code(),
            StatusCode::kInvalidArgument);
  ASSERT_TRUE(registry.register_pinned(60, registry_digest(9), 0, 4096).ok());
  EXPECT_EQ(registry.register_allocation(60, registry_digest(9), 8192).code(),
            StatusCode::kFailedPrecondition);
}

TEST(EngineCudaOwnerLedgerRegistryTest,
     ReportsMissingRequestedIdentityAsUnavailableWithoutMutation) {
  RegistryClock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  const std::array<std::uint64_t, 1> pinned{60};
  const std::array<std::uint64_t, 1> allocations{70};
  EXPECT_EQ(registry.capture(pinned, allocations).status().code(),
            StatusCode::kUnavailable);
  ASSERT_TRUE(registry.register_pinned(60, registry_digest(9), 0, 4096).ok());
  ASSERT_TRUE(registry.register_allocation(70, registry_digest(9), 8192).ok());
  EXPECT_TRUE(registry.capture(pinned, allocations).ok());
}

TEST(EngineCudaOwnerLedgerRegistryTest, RejectsClockAndSampleExhaustion) {
  RegistryClock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  ASSERT_TRUE(registry.register_pinned(60, registry_digest(9), 0, 4096).ok());
  ASSERT_TRUE(registry.register_allocation(70, registry_digest(9), 8192).ok());
  const std::array<std::uint64_t, 1> pinned{60};
  const std::array<std::uint64_t, 1> allocations{70};
  clock.values[1] = 99;
  EXPECT_EQ(registry.capture(pinned, allocations).status().code(),
            StatusCode::kFailedPrecondition);
}

TEST(EngineCudaOwnerLedgerRegistryTest, PoisonPermanentlyRejectsCapture) {
  RegistryClock clock;
  EngineCudaOwnerLedgerRegistry registry(clock);
  ASSERT_TRUE(registry.register_pinned(60, registry_digest(9), 0, 4096).ok());
  ASSERT_TRUE(registry.register_allocation(70, registry_digest(9), 8192).ok());
  registry.poison("allocation teardown drifted");
  const std::array<std::uint64_t, 1> pinned{60};
  const std::array<std::uint64_t, 1> allocations{70};
  EXPECT_EQ(registry.capture(pinned, allocations).status().code(),
            StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
