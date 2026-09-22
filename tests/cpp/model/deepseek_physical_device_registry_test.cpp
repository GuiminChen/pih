#include "pih/model/deepseek_physical_device_registry.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest commitment(std::byte value) {
  Sha256Digest result{}; result.bytes.fill(value); return result;
}

TEST(DeepSeekPhysicalDeviceRegistryTest, FreezesOneToFourCanonicalDevices) {
  for (std::uint32_t count = 1; count <= 4; ++count) {
    std::vector<DeepSeekPhysicalDeviceRegistryEntry> entries;
    for (std::uint32_t rank = 0; rank < count; ++rank)
      entries.push_back({rank, static_cast<std::int32_t>(rank * 2U + 1U),
                         100U + rank,
                         commitment(static_cast<std::byte>(rank + 1))});
    auto registry = DeepSeekPhysicalDeviceRegistry::Create(entries);
    ASSERT_TRUE(registry.ok()) << registry.status().message();
    EXPECT_EQ(registry->world_size(), count);
    EXPECT_EQ(registry->rank(count - 1)->registry_identity, 99U + count);
    EXPECT_EQ(registry->identity(100)->rank, 0U);
  }
}

TEST(DeepSeekPhysicalDeviceRegistryTest, RejectsDuplicateGpuOrRegistryIdentity) {
  std::array<DeepSeekPhysicalDeviceRegistryEntry, 2> entries{{
      {0, 2, 100, commitment(std::byte{1})},
      {1, 5, 101, commitment(std::byte{1})}}};
  EXPECT_FALSE(DeepSeekPhysicalDeviceRegistry::Create(entries).ok());
  entries[1].uuid_commitment = commitment(std::byte{2});
  entries[1].registry_identity = 100;
  EXPECT_FALSE(DeepSeekPhysicalDeviceRegistry::Create(entries).ok());
}

TEST(DeepSeekPhysicalDeviceRegistryTest, RejectsZeroCommitmentAndRankGap) {
  std::array<DeepSeekPhysicalDeviceRegistryEntry, 1> entries{{{0, 2, 100, {}}}};
  EXPECT_FALSE(DeepSeekPhysicalDeviceRegistry::Create(entries).ok());
  entries[0] = {1, 2, 100, commitment(std::byte{1})};
  EXPECT_FALSE(DeepSeekPhysicalDeviceRegistry::Create(entries).ok());
}

TEST(DeepSeekPhysicalDeviceRegistryTest, RejectsDuplicateStartupOrdinal) {
  std::array<DeepSeekPhysicalDeviceRegistryEntry, 2> entries{{
      {0, 2, 100, commitment(std::byte{1})},
      {1, 2, 101, commitment(std::byte{2})}}};
  EXPECT_FALSE(DeepSeekPhysicalDeviceRegistry::Create(entries).ok());
}

}  // namespace
}  // namespace pih
