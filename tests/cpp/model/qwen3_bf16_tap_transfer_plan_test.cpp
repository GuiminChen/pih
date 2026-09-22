#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_tap_transfer_plan.h"

namespace pih {
namespace {

class DeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    return Allocation{reinterpret_cast<void*>(UINT64_C(0x400000000)), bytes,
                      alignment, 41,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation) noexcept override {}
};

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, 73, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
};

class Placement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void*, std::uint64_t, std::int32_t numa) override {
    return numa == 2 ? Status::Ok()
                     : Status::FailedPrecondition("wrong NUMA node");
  }
};

QwenBf16TapTransferIdentity identity() {
  return {7, 90, 91, 92, 501, 502, 99, 123, 124, 200, 300, 0};
}

TEST(QwenBf16TapTransferPlanTest, CompilesMirroredD2dAndD2hTransfers) {
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 0, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
  DeviceAllocator device_allocator;
  PinnedAllocator pinned_allocator;
  Placement placement;
  auto arenas = QwenNumericalTapArenas::AllocateVerified(
                    taps, device_allocator, pinned_allocator, placement, 0, 2)
                    .value();
  const std::array<CudaCopyEndpoint, 2> endpoints{{
      {0x500000000, 4096, 0, 10, 11, CudaCopyMemoryType::kDevice, 0, 0},
      {0x600000000, 151936 * 4, 0, 12, 13,
       CudaCopyMemoryType::kDevice, 0, 0},
  }};
  auto plan = QwenBf16TapTransferPlan::Create(taps, endpoints, arenas,
                                               identity());

  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->size(), 2);
  EXPECT_EQ(plan->identity().execution_stream, 123);
  EXPECT_EQ(plan->identity().diagnostic_stream, 124);
  for (const auto& transfer : plan->transfers()) {
    EXPECT_EQ(transfer.state(),
              QwenNumericalTapTransferState::kProducerPending);
    EXPECT_EQ(transfer.producer_plan_generation(), 90);
    EXPECT_EQ(transfer.snapshot_stream(), 123);
  }
}

TEST(QwenBf16TapTransferPlanTest, RejectsCrossRankAndEventIdentityDrift) {
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 0, 1}};
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
  DeviceAllocator device_allocator;
  PinnedAllocator pinned_allocator;
  Placement placement;
  auto arenas = QwenNumericalTapArenas::AllocateVerified(
                    taps, device_allocator, pinned_allocator, placement, 0, 2)
                    .value();
  const std::array<CudaCopyEndpoint, 1> wrong_rank{{
      {0x500000000, 4096, 0, 10, 11, CudaCopyMemoryType::kDevice, 1, 1},
  }};
  EXPECT_FALSE(QwenBf16TapTransferPlan::Create(
                   taps, wrong_rank, arenas, identity())
                   .ok());
  auto same_event = identity();
  same_event.host_event_generation = same_event.snapshot_event_generation;
  const std::array<CudaCopyEndpoint, 1> source{{
      {0x500000000, 4096, 0, 10, 11, CudaCopyMemoryType::kDevice, 0, 0},
  }};
  EXPECT_FALSE(QwenBf16TapTransferPlan::Create(
                   taps, source, arenas, same_event)
                   .ok());
}

}  // namespace
}  // namespace pih
