#include "pih/model/qwen3_bf16_engine_arenas.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class VirtualCudaAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    if (calls == fail_on_call) {
      return Status::ResourceExhausted("injected allocation failure");
    }
    const auto address = UINT64_C(0x100000000) +
                         static_cast<std::uint64_t>(calls) * UINT64_C(0x100000000);
    return Allocation{reinterpret_cast<void*>(address), bytes, alignment,
                      static_cast<std::uint64_t>(calls),
                      Device::Create(DeviceType::kCuda, device).value()};
  }
  void deallocate(Allocation) noexcept override { ++releases; }
  int calls = 0;
  int releases = 0;
  int fail_on_call = 0;
  int device = 0;
};

TEST(QwenBf16EngineDeviceArenasTest, AtomicallyOwnsEveryPlannedArena) {
  auto plan = QwenBf16EngineResourcePlan::Create(17, 32, 2, 4096).value();
  VirtualCudaAllocator allocator;
  {
    auto arenas = QwenBf16EngineDeviceArenas::Allocate(plan, allocator, 0);
    ASSERT_TRUE(arenas.ok()) << arenas.status().message();
    EXPECT_EQ(allocator.calls, 10);
    auto owners = arenas->step_owners();
    EXPECT_EQ(owners.step_staging.bytes, plan.step_staging_bytes());
    EXPECT_EQ(owners.activations.bytes, plan.activation_bytes());
    EXPECT_EQ(owners.kv_backing.bytes, plan.kv_backing_bytes());
    EXPECT_EQ(owners.device_error.generation, 7);
    EXPECT_EQ(arenas->linear_workspace_bytes(), 4096);
  }
  EXPECT_EQ(allocator.releases, 10);
}

TEST(QwenBf16EngineDeviceArenasTest, PartialFailureRollsBackEveryPriorArena) {
  auto plan = QwenBf16EngineResourcePlan::Create(17, 32, 2, 4096).value();
  VirtualCudaAllocator allocator;
  allocator.fail_on_call = 8;
  EXPECT_FALSE(QwenBf16EngineDeviceArenas::Allocate(plan, allocator, 0).ok());
  EXPECT_EQ(allocator.calls, 8);
  EXPECT_EQ(allocator.releases, 7);
}

TEST(QwenBf16EngineDeviceArenasTest, RejectsAllocatorOnWrongDevice) {
  auto plan = QwenBf16EngineResourcePlan::Create(1, 16, 1, 0).value();
  VirtualCudaAllocator allocator;
  allocator.device = 1;
  EXPECT_FALSE(QwenBf16EngineDeviceArenas::Allocate(plan, allocator, 0).ok());
  EXPECT_EQ(allocator.releases, 10);
}

}  // namespace
}  // namespace pih
