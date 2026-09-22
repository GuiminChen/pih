#include "pih/model/deepseek_gpu_uuid_commitment.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekGpuUuidCommitmentTest, CommitsAllSixteenUuidBytes) {
  std::array<std::byte, 16> first{};
  std::array<std::byte, 16> second{};
  first[0] = std::byte{1}; second[0] = std::byte{1};
  second[15] = std::byte{2};
  auto a = deepseek_gpu_uuid_commitment(first);
  auto b = deepseek_gpu_uuid_commitment(second);
  ASSERT_TRUE(a.ok()); ASSERT_TRUE(b.ok());
  EXPECT_NE(*a, *b);
  EXPECT_NE(*a, Sha256Digest{});
}

TEST(DeepSeekGpuUuidCommitmentTest, RejectsZeroOrWrongSizedUuid) {
  std::array<std::byte, 16> zero{};
  EXPECT_FALSE(deepseek_gpu_uuid_commitment(zero).ok());
  EXPECT_FALSE(deepseek_gpu_uuid_commitment(
      std::span<const std::byte>(zero).first(15)).ok());
}

}  // namespace
}  // namespace pih
