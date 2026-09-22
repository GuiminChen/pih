#include "pih/model/deepseek_expert_compute_lane.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class PinnedHeap final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    void* p = _aligned_malloc(static_cast<std::size_t>(bytes),
                              static_cast<std::size_t>(alignment));
    if (p == nullptr) return Status::ResourceExhausted("allocation failed");
    return Allocation{p, bytes, alignment, 9, Device::Cpu()};
  }
  void deallocate(Allocation a) noexcept override { _aligned_free(a.data); }
};

class ImmediateOps final : public DeepSeekExpertCudaOperations {
 public:
  Status validate_host_staging(const DeepSeekExpertHostStaging&) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return hit(); }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t, std::uintptr_t) override { return hit(); }
  Status copy_d2h_async(void* host, std::uintptr_t, std::size_t, std::uintptr_t) override {
    *static_cast<std::uint32_t*>(host)=0; return hit();
  }
  Status gather(DeepSeekRouteGatherLaunch) override { return hit(); }
  Status quantize(DeepSeekFp8ActivationQuantLaunch) override { return hit(); }
  Status gemm(DeepSeekFp4GemmLaunch) override { return hit(); }
  Status swiglu(DeepSeekExpertSwiGluLaunch) override { return hit(); }
  Status accumulate(DeepSeekExpertAccumulateLaunch) override { return hit(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return hit(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  Status hit() { ++calls; return Status::Ok(); }
  int calls=0;
};

TEST(DeepSeekExpertComputeLaneTest, KeepsBackendPointersStableAcrossMove) {
  ImmediateOps ops; PinnedHeap allocator;
  auto slots=DeepSeekExpertSlotTable::Create({0x10000000,0x11000000},77,0);
  auto layout=DeepSeekExpertComputeArenaLayout::Create(1);
  ASSERT_TRUE(slots.ok()); ASSERT_TRUE(layout.ok());
  auto arena=layout->bind(0x300000,layout->required_bytes()); ASSERT_TRUE(arena.ok());
  auto lane=DeepSeekExpertComputeLane::Create(
      ops,allocator,std::move(*slots),*arena,1,0x400000,0x500000,0x600000,
      0x700000,77);
  ASSERT_TRUE(lane.ok());
  DeepSeekExpertComputeLane moved=std::move(*lane);

  auto pager=DeepSeekExpertPager::Create({4,4},2,2); ASSERT_TRUE(pager.ok());
  auto demand=pager->demand({4,3}); ASSERT_TRUE(demand.ok());
  ASSERT_TRUE(pager->begin_h2d({4,3},demand->generation,
                               DeepSeekExpertPager::kBundleBytes).ok());
  ASSERT_TRUE(pager->complete_h2d({4,3},demand->generation).ok());
  auto lease=pager->acquire({4,3},demand->generation); ASSERT_TRUE(lease.ok());
  DeepSeekExpertRoute route{3,0,0,1.0F};
  ASSERT_TRUE(moved.kernel().launch(*lease,&route,1).ok());
  auto result=moved.kernel().poll(); ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result,DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_EQ(ops.calls,13);
}

TEST(DeepSeekExpertComputeLaneTest, RejectsBeforeAllocatingForInvalidTokenCount) {
  ImmediateOps ops; PinnedHeap allocator;
  auto slots=DeepSeekExpertSlotTable::Create({0x10000000,0x11000000},77,0);
  ASSERT_TRUE(slots.ok());
  DeepSeekExpertComputeArena arena{};
  EXPECT_FALSE(DeepSeekExpertComputeLane::Create(
      ops,allocator,std::move(*slots),arena,0,1,2,3,4,77).ok());
}

TEST(DeepSeekExpertComputeLaneTest,
     CreatesDedicatedDsparkLaneOnlyWhenSourceIsBound) {
  ImmediateOps ops; PinnedHeap allocator;
  auto slots = DeepSeekExpertSlotTable::CreateResidentOnly(77, 0);
  auto layout = DeepSeekExpertComputeArenaLayout::Create(5);
  ASSERT_TRUE(slots.ok()); ASSERT_TRUE(layout.ok());
  auto arena = layout->bind(0x300000, layout->required_bytes());
  ASSERT_TRUE(arena.ok());
  auto lane = DeepSeekExpertComputeLane::Create(
      ops, allocator, std::move(*slots), *arena, 5, 0x400000,
      0x500000, 0x600000, 0x700000, 77, 0x800000);
  ASSERT_TRUE(lane.ok());
  EXPECT_NE(lane->dspark_kernel(), nullptr);
}

} }  // namespace pih
