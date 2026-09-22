#include "pih/model/engine_physical_gpu_uuid_commitment.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(EnginePhysicalGpuUuidCommitmentTest, UsesFrozenEngineDomainAndAllUuidBytes) {
  std::array<std::byte, 16> uuid{};
  for (std::size_t index = 0; index < uuid.size(); ++index)
    uuid[index] = static_cast<std::byte>(index + 1);
  auto digest = engine_physical_gpu_uuid_commitment(uuid);
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest->hex(),
            "4b0353f5d469a24368ce25f29acc840f52f2a4152c6ab97b96b7a31e50e6ce8c");
}

TEST(EnginePhysicalGpuUuidCommitmentTest, RejectsZeroOrWrongSizedUuid) {
  std::array<std::byte, 16> zero{};
  EXPECT_FALSE(engine_physical_gpu_uuid_commitment(zero).ok());
  EXPECT_FALSE(engine_physical_gpu_uuid_commitment(
                   std::span<const std::byte>(zero).first(15)).ok());
}

}  // namespace
}  // namespace pih
