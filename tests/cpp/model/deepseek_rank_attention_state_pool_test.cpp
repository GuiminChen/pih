#include "pih/model/deepseek_rank_attention_state_pool.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

class PoolAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    allocations.push_back({bytes, alignment});
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("pool allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> allocations;
 private:
  std::uint64_t generation_ = 0;
};

class PoolFixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

TEST(DeepSeekRankAttentionStatePoolTest,
     ComputesPerLayerPersistentAndTailCowCredits) {
  PoolAllocator allocator;
  PoolFixedOperations operations;
  const DeepSeekStagePlan stage{0, {2, 3}, false, false, false};
  auto pool = DeepSeekRankAttentionStatePool::Allocate(
      allocator, stage, 2, 256, 19, operations, 17, 0);
  ASSERT_TRUE(pool.ok()) << pool.status().message();
  EXPECT_EQ(pool->sequence_capacity(), 2U);
  EXPECT_EQ(pool->reserved_tokens_per_sequence(), 256U);
  // Layer 2: ceil((256/4)/64) persistent + one COW.
  EXPECT_EQ(pool->ratio4_page_pairs_per_sequence(), 2U);
  // Layer 3: ceil((256/128)/64) persistent + one COW.
  EXPECT_EQ(pool->ratio128_pages_per_sequence(), 2U);
  EXPECT_EQ(pool->ratio4_pool().free_pairs(), 4U);
  EXPECT_EQ(pool->ratio128_pool().free_pages(), 4U);
  ASSERT_FALSE(allocator.allocations.empty());
  EXPECT_EQ(allocator.allocations.front().first,
            pool->physical_layout().allocation_bytes());
  EXPECT_EQ(allocator.allocations.front().second, 65536U);
  const auto arena = pool->page_arena();
  const auto& layout = pool->physical_layout();
  EXPECT_EQ(
      arena.ratio4_index_base() - arena.ratio4_main_base(),
      layout.extent(DeepSeekAttentionPhysicalExtentKind::kRatio4Index)
              .offset_bytes -
          layout.extent(DeepSeekAttentionPhysicalExtentKind::kRatio4Main)
              .offset_bytes);
  EXPECT_EQ(
      arena.ratio128_main_base() - arena.ratio4_main_base(),
      layout.extent(DeepSeekAttentionPhysicalExtentKind::kRatio128Main)
              .offset_bytes -
          layout.extent(DeepSeekAttentionPhysicalExtentKind::kRatio4Main)
              .offset_bytes);
  auto first = pool->ratio4_pool().reserve(7, 2, 0).value();
  auto second = pool->ratio4_pool().reserve(8, 2, 0).value();
  EXPECT_NE(first.main.slot(), second.main.slot());
  EXPECT_TRUE(pool->page_arena().Resolve(first.main).ok());
  EXPECT_NE(pool->sequence(0).fixed_banks().committed_address(),
            pool->sequence(1).fixed_banks().committed_address());
}

TEST(DeepSeekRankAttentionStatePoolTest,
     RejectsStageWithoutBothCompressedKinds) {
  PoolAllocator allocator;
  PoolFixedOperations operations;
  EXPECT_FALSE(DeepSeekRankAttentionStatePool::Allocate(
      allocator, {0, {0, 1}, true, false, false}, 1, 256, 19,
      operations, 17, 0).ok());
}

}  // namespace
}  // namespace pih
