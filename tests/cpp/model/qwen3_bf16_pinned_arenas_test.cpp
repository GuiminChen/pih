#include "pih/model/qwen3_bf16_pinned_arenas.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class FakePinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    if (calls == fail_on_call) {
      return Status::ResourceExhausted("injected pinned failure");
    }
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment,
                      static_cast<std::uint64_t>(calls), Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
  int calls = 0;
  int releases = 0;
  int fail_on_call = 0;
};

class Placement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void* data, std::uint64_t bytes,
                std::int32_t numa_node) override {
    ++calls;
    if (data == nullptr || bytes == 0 || numa_node != 1) {
      return Status::FailedPrecondition("unexpected placement");
    }
    return status;
  }
  int calls = 0;
  Status status = Status::Ok();
};

TEST(QwenBf16PinnedHostArenasTest, OwnsTypedDmaEndpoints) {
  auto plan = QwenBf16EngineResourcePlan::Create(17, 32, 2, 0).value();
  FakePinnedAllocator allocator;
  Placement placement;
  {
    auto arenas = QwenBf16PinnedHostArenas::AllocateVerified(
        plan, allocator, placement, 1);
    ASSERT_TRUE(arenas.ok()) << arenas.status().message();
    EXPECT_EQ(arenas->staging().size(), plan.step_staging_bytes());
    EXPECT_EQ(arenas->result().size(), plan.pinned_result_bytes());
    auto staging = arenas->staging_endpoint(11, 0).value();
    auto result = arenas->result_endpoint(12, 0).value();
    EXPECT_EQ(staging.memory_type,
              CudaCopyMemoryType::kRegisteredPinnedHost);
    EXPECT_EQ(staging.allocation_base,
              reinterpret_cast<std::uintptr_t>(arenas->staging().data()));
    EXPECT_NE(staging.owner_id, result.owner_id);
  }
  EXPECT_EQ(placement.calls, 2);
  EXPECT_EQ(allocator.releases, 2);
}

TEST(QwenBf16PinnedHostArenasTest, UnverifiedArenaCannotPublishEndpoint) {
  auto plan = QwenBf16EngineResourcePlan::Create(1, 16, 1, 0).value();
  FakePinnedAllocator allocator;
  auto arenas = QwenBf16PinnedHostArenas::Allocate(plan, allocator).value();
  EXPECT_FALSE(arenas.staging_endpoint(11, 0).ok());
  EXPECT_FALSE(arenas.result_endpoint(12, 0).ok());
}

TEST(QwenBf16PinnedHostArenasTest, VerificationFailureRollsBackBothArenas) {
  auto plan = QwenBf16EngineResourcePlan::Create(1, 16, 1, 0).value();
  FakePinnedAllocator allocator;
  Placement placement;
  placement.status = Status::FailedPrecondition("wrong NUMA node");
  EXPECT_FALSE(QwenBf16PinnedHostArenas::AllocateVerified(
      plan, allocator, placement, 1).ok());
  EXPECT_EQ(allocator.releases, 2);
}

TEST(QwenBf16PinnedHostArenasTest, FailureRollsBackPriorAllocation) {
  auto plan = QwenBf16EngineResourcePlan::Create(1, 16, 1, 0).value();
  FakePinnedAllocator allocator;
  allocator.fail_on_call = 2;
  EXPECT_FALSE(QwenBf16PinnedHostArenas::Allocate(plan, allocator).ok());
  EXPECT_EQ(allocator.releases, 1);
}

}  // namespace
}  // namespace pih
