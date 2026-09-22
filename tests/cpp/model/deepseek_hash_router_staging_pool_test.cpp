#include "pih/model/deepseek_hash_router_staging_pool.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class HashStagingAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("hash staging");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekHashRouterStagingPoolTest,
     LeasesEveryOwnedHashLayerUntilPlanBackingReleases) {
  HashStagingAllocator allocator;
  auto pool = DeepSeekHashRouterStagingPool::Allocate(
      {1, 4}, 8, allocator).value();
  EXPECT_EQ(pool.layer_count(), 2U);
  {
    auto leased = pool.acquire();
    ASSERT_TRUE(leased.ok());
    ASSERT_EQ(leased->size(), 2U);
    EXPECT_EQ((*leased)[0].layer, 1U);
    EXPECT_EQ((*leased)[1].layer, 2U);
    EXPECT_FALSE(pool.acquire().ok());
  }
  EXPECT_TRUE(pool.acquire().ok());
}

TEST(DeepSeekHashRouterStagingPoolTest,
     LearnedOnlyStageAllocatesNoHashStorage) {
  HashStagingAllocator allocator;
  auto pool = DeepSeekHashRouterStagingPool::Allocate(
      {3, 42}, 8, allocator).value();
  EXPECT_EQ(pool.layer_count(), 0U);
  EXPECT_TRUE(pool.acquire()->empty());
}

}}  // namespace pih::<anonymous>
