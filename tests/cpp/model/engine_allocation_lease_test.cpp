#include "pih/model/engine_allocation_lease.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest deployment() {
  Sha256Digest value{}; value.bytes.fill(std::byte{3}); return value;
}

EngineAllocationLeaseManifest manifest() { return {deployment(), {10, 20}}; }

TEST(EngineAllocationLeaseTest, DigestsExactSortedPhysicalGpuSet) {
  auto digest = engine_allocation_lease_digest(manifest());
  ASSERT_TRUE(digest.ok());
  auto changed = manifest(); changed.sorted_physical_gpu_identities[1] = 30;
  ASSERT_TRUE(engine_allocation_lease_digest(changed).ok());
  EXPECT_NE(*digest, *engine_allocation_lease_digest(changed));
  EXPECT_EQ(*digest, *engine_allocation_lease_digest(manifest()));
}

TEST(EngineAllocationLeaseTest, RejectsInvalidOrNoncanonicalManifest) {
  EXPECT_FALSE(engine_allocation_lease_digest({{}, {10}}).ok());
  EXPECT_FALSE(engine_allocation_lease_digest({deployment(), {}}).ok());
  EXPECT_FALSE(engine_allocation_lease_digest(
      {deployment(), {1, 2, 3, 4, 5}}).ok());
  EXPECT_FALSE(engine_allocation_lease_digest({deployment(), {20, 10}}).ok());
  EXPECT_FALSE(engine_allocation_lease_digest({deployment(), {10, 10}}).ok());
  EXPECT_FALSE(engine_allocation_lease_digest({deployment(), {0}}).ok());
}

TEST(EngineAllocationLeaseTest, VerifiesDescriptorTokenAndExclusiveLock) {
  const auto digest = engine_allocation_lease_digest(manifest()).value();
  EngineAllocationLeaseExpectation expected{digest, 7, 8};
  EngineAllocationLeaseObservation observed{digest, 7, 8, true, true};
  EXPECT_TRUE(verify_engine_allocation_lease(expected, observed).ok());
  for (int mutation = 0; mutation < 5; ++mutation) {
    auto value = observed;
    if (mutation == 0) value.token_digest.bytes[0] = std::byte{9};
    if (mutation == 1) ++value.filesystem_identity;
    if (mutation == 2) ++value.file_identity;
    if (mutation == 3) value.descriptor_open = false;
    if (mutation == 4) value.exclusive_ofd_lock_held = false;
    EXPECT_FALSE(verify_engine_allocation_lease(expected, value).ok())
        << mutation;
  }
}

TEST(EngineAllocationLeaseTest, RejectsIncompleteExpectation) {
  const auto digest = engine_allocation_lease_digest(manifest()).value();
  EngineAllocationLeaseObservation observed{digest, 7, 8, true, true};
  EXPECT_FALSE(verify_engine_allocation_lease({}, observed).ok());
  EXPECT_FALSE(verify_engine_allocation_lease({digest, 0, 8}, observed).ok());
  EXPECT_FALSE(verify_engine_allocation_lease({digest, 7, 0}, observed).ok());
}

}  // namespace
}  // namespace pih
