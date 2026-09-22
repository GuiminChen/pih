#include "pih/backend/cuda/deepseek_expert_host_staging.h"
#include "pih/model/deepseek_expert_bundle_staging_pool.h"

#include <cstdlib>

#include <gtest/gtest.h>

namespace pih { namespace {

class PinnedHeap final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    ++allocations;
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("allocation failed");
    return Allocation{data, bytes, alignment, 41, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++deallocations; _aligned_free(allocation.data);
  }
  int allocations = 0;
  int deallocations = 0;
};

TEST(DeepSeekExpertHostStagingTest, OwnsOneAlignedAllocationAndTypedSlices) {
  PinnedHeap allocator;
  {
    auto owner = DeepSeekExpertHostStagingOwner::Allocate(4096, allocator);
    ASSERT_TRUE(owner.ok());
    auto staging = owner->staging();
    EXPECT_EQ(allocator.allocations, 1);
    EXPECT_EQ(staging.capacity, 4096U);
    EXPECT_NE(staging.route_weights, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(staging.route_weights) % 256, 0U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(staging.token_indices) % 256, 0U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(staging.error_flag) % 256, 0U);
    EXPECT_LT(reinterpret_cast<std::uintptr_t>(staging.route_weights),
              reinterpret_cast<std::uintptr_t>(staging.token_indices));
    EXPECT_LT(reinterpret_cast<std::uintptr_t>(staging.token_indices),
              reinterpret_cast<std::uintptr_t>(staging.error_flag));
    EXPECT_EQ(owner->generation(), 41U);
  }
  EXPECT_EQ(allocator.deallocations, 1);
}

TEST(DeepSeekExpertHostStagingTest, RejectsCapacityOutsideSealedArenaLimit) {
  PinnedHeap allocator;
  EXPECT_FALSE(DeepSeekExpertHostStagingOwner::Allocate(0, allocator).ok());
  EXPECT_FALSE(DeepSeekExpertHostStagingOwner::Allocate(
      DeepSeekExpertComputeArenaLayout::kMaximumTokens + 1, allocator).ok());
  EXPECT_EQ(allocator.allocations, 0);
}

TEST(DeepSeekExpertHostStagingTest, OwnsCanonicalBundleExtentsInOneRegistration) {
  PinnedHeap allocator;
  {
    auto pool = DeepSeekExpertBundleStagingPool::Allocate(2, allocator);
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    EXPECT_EQ(pool->extent_count(), 2U);
    EXPECT_EQ(pool->bytes(), 2U * DeepSeekExpertBundleLayout::kBundleBytes);
    auto extents = pool->extents();
    ASSERT_EQ(extents.size(), 2U);
    EXPECT_EQ(extents[0].bytes, DeepSeekExpertBundleLayout::kBundleBytes);
    EXPECT_EQ(extents[1].address - extents[0].address,
              DeepSeekExpertBundleLayout::kBundleBytes);
    EXPECT_EQ(extents[0].registration_identity,
              extents[1].registration_identity);
    EXPECT_EQ(extents[0].registration_identity, 41U);
  }
  EXPECT_EQ(allocator.allocations, 1);
  EXPECT_EQ(allocator.deallocations, 1);
}

TEST(DeepSeekExpertHostStagingTest, RejectsBundlePoolOutsideBoundedEnvelope) {
  PinnedHeap allocator;
  EXPECT_FALSE(DeepSeekExpertBundleStagingPool::Allocate(1, allocator).ok());
  EXPECT_FALSE(DeepSeekExpertBundleStagingPool::Allocate(
      DeepSeekExpertBundleStagingPool::kMaximumExtentCount + 1U,
      allocator).ok());
  EXPECT_EQ(allocator.allocations, 0);
}

} }  // namespace pih
