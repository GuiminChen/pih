#include "pih/model/deepseek_rank_process_manifest_plan.h"

#include <gtest/gtest.h>

#include <limits>

namespace pih {
namespace {

Sha256Digest commitment(std::byte value) {
  Sha256Digest result{}; result.bytes.fill(value); return result;
}

TEST(DeepSeekRankProcessManifestPlanTest, ProjectsFrozenDevicesToCanonicalRanks) {
  const std::array<DeepSeekPhysicalDeviceRegistryEntry, 3> entries{{
      {0, 2, 10, commitment(std::byte{1})},
      {1, 5, 11, commitment(std::byte{2})},
      {2, 9, 12, commitment(std::byte{3})}}};
  auto devices = DeepSeekPhysicalDeviceRegistry::Create(entries).value();
  auto plan = DeepSeekRankProcessManifestPlan::Create(7, 8, 100, 200, devices);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->manifests().size(), 3U);
  for (std::uint32_t rank = 0; rank < 3; ++rank) {
    const auto& manifest = plan->manifests()[rank];
    EXPECT_EQ(manifest.rank, rank);
    EXPECT_EQ(manifest.world_size, 3U);
    EXPECT_EQ(manifest.physical_device_identity, 10U + rank);
    EXPECT_EQ(manifest.process_manifest_identity, 100U + rank);
    EXPECT_EQ(manifest.physical_device_uuid_commitment,
              entries[rank].uuid_commitment);
    EXPECT_EQ(manifest.startup_device_ordinal,
              entries[rank].startup_device_ordinal);
    EXPECT_EQ(manifest.startup_deadline_ns, 200U);
  }
}

TEST(DeepSeekRankProcessManifestPlanTest, RejectsIdentityOverflow) {
  const std::array<DeepSeekPhysicalDeviceRegistryEntry, 2> entries{{
      {0, 2, 10, commitment(std::byte{1})},
      {1, 5, 11, commitment(std::byte{2})}}};
  auto devices = DeepSeekPhysicalDeviceRegistry::Create(entries).value();
  EXPECT_FALSE(DeepSeekRankProcessManifestPlan::Create(
      7, 8, std::numeric_limits<std::uint64_t>::max(), 200, devices).ok());
  EXPECT_FALSE(DeepSeekRankProcessManifestPlan::Create(0, 8, 100, 200, devices).ok());
  EXPECT_FALSE(DeepSeekRankProcessManifestPlan::Create(7, 8, 100, 0, devices).ok());
}

}  // namespace
}  // namespace pih
