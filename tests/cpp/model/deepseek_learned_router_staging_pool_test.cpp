#include "pih/model/deepseek_learned_router_staging_pool.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class StagingAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("staging allocation");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekLearnedRouterStagingPoolTest,
     LeasesEveryOwnedLearnedLayerUntilPlanBackingReleases) {
  StagingAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate(
      {2, 4}, 8, allocator).value();
  EXPECT_EQ(pool.layer_count(), 2U);
  {
    auto leased = pool.acquire();
    ASSERT_TRUE(leased.ok()) << leased.status().message();
    ASSERT_EQ(leased->size(), 2U);
    EXPECT_EQ((*leased)[0].layer, 3U);
    EXPECT_EQ((*leased)[1].layer, 4U);
    EXPECT_NE((*leased)[0].staging->generation(),
              (*leased)[1].staging->generation());
    EXPECT_FALSE(pool.acquire().ok());
  }
  EXPECT_TRUE(pool.acquire().ok());
}

TEST(DeepSeekLearnedRouterStagingPoolTest,
     HashOnlyStageAllocatesNoPinnedRouterStorage) {
  StagingAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate(
      {0, 2}, 8, allocator).value();
  EXPECT_EQ(pool.layer_count(), 0U);
  EXPECT_TRUE(pool.acquire()->empty());
}

TEST(DeepSeekLearnedRouterStagingPoolTest,
     AllocatesPinnedStagingOnlyForMainLayers) {
  StagingAllocator allocator;
  auto pool = DeepSeekLearnedRouterStagingPool::Allocate(
      {41, 42}, 8, allocator);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  EXPECT_EQ(pool->layer_count(), 2U);
  auto leased = pool->acquire();
  ASSERT_TRUE(leased.ok());
  ASSERT_EQ(leased->size(), 2U);
  EXPECT_EQ(leased->back().layer, 42U);
  EXPECT_FALSE(DeepSeekLearnedRouterStagingPool::Allocate(
      {43, 43}, 8, allocator).ok());
}

}  // namespace
}  // namespace pih
