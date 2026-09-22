#include <cstddef>
#include <cstdint>
#include <new>

#include <gtest/gtest.h>

#include "pih/model/qwen3_numerical_tap_arenas.h"

namespace pih {
namespace {

class TapDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    if (fail) return Status::ResourceExhausted("device failure");
    return Allocation{reinterpret_cast<void*>(UINT64_C(0x400000000)), bytes,
                      alignment, 41,
                      Device::Create(DeviceType::kCuda, device).value()};
  }
  void deallocate(Allocation) noexcept override { ++releases; }
  int device = 0;
  int calls = 0;
  int releases = 0;
  bool fail = false;
};

class TapPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    if (fail) return Status::ResourceExhausted("pinned failure");
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, 73, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
  int calls = 0;
  int releases = 0;
  bool fail = false;
};

class TapPlacement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void* data, std::uint64_t bytes,
                std::int32_t numa_node) override {
    ++calls;
    if (data == nullptr || bytes == 0 || numa_node != 2) {
      return Status::FailedPrecondition("unexpected placement input");
    }
    return result;
  }
  int calls = 0;
  Status result = Status::Ok();
};

QwenNumericalTapPlan tap_plan() {
  const QwenNumericalTapRequest requests[]{
      {QwenNumericalTapPoint::kLayerHidden, 0, 0, 1},
      {QwenNumericalTapPoint::kLogits, 28, 0, 1},
  };
  return QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
}

TEST(QwenNumericalTapArenasTest, OwnsMirroredTypedCaptureRanges) {
  const auto plan = tap_plan();
  TapDeviceAllocator device;
  TapPinnedAllocator pinned;
  TapPlacement placement;
  {
    auto arenas = QwenNumericalTapArenas::AllocateVerified(
        plan, device, pinned, placement, 0, 2);
    ASSERT_TRUE(arenas.ok()) << arenas.status().message();
    EXPECT_EQ(arenas->arena_bytes(), plan.arena_bytes());
    auto snapshot = arenas->device_endpoint(1, 1001, 0).value();
    auto host = arenas->pinned_endpoint(1, 1002, 0).value();
    EXPECT_EQ(snapshot.memory_type, CudaCopyMemoryType::kDevice);
    EXPECT_EQ(snapshot.generation, 41);
    EXPECT_EQ(host.memory_type,
              CudaCopyMemoryType::kRegisteredPinnedHost);
    EXPECT_EQ(host.generation, 73);
    EXPECT_EQ(snapshot.offset, plan.captures()[1].offset_bytes);
    EXPECT_EQ(host.offset, snapshot.offset);
    EXPECT_EQ(arenas->pinned_capture(1).value().size(),
              plan.captures()[1].size_bytes);
  }
  EXPECT_EQ(placement.calls, 1);
  EXPECT_EQ(device.releases, 1);
  EXPECT_EQ(pinned.releases, 1);
}

TEST(QwenNumericalTapArenasTest, AllocationOrPlacementFailureRollsBack) {
  const auto plan = tap_plan();
  TapDeviceAllocator device;
  TapPinnedAllocator pinned;
  TapPlacement placement;
  pinned.fail = true;
  EXPECT_FALSE(QwenNumericalTapArenas::AllocateVerified(
                   plan, device, pinned, placement, 0, 2)
                   .ok());
  EXPECT_EQ(device.releases, 1);

  pinned.fail = false;
  placement.result = Status::FailedPrecondition("wrong NUMA node");
  EXPECT_FALSE(QwenNumericalTapArenas::AllocateVerified(
                   plan, device, pinned, placement, 0, 2)
                   .ok());
  EXPECT_EQ(device.releases, 2);
  EXPECT_EQ(pinned.releases, 1);
}

TEST(QwenNumericalTapArenasTest, RejectsWrongDeviceAndCaptureIdentity) {
  const auto plan = tap_plan();
  TapDeviceAllocator device;
  device.device = 1;
  TapPinnedAllocator pinned;
  TapPlacement placement;
  EXPECT_FALSE(QwenNumericalTapArenas::AllocateVerified(
                   plan, device, pinned, placement, 0, 2)
                   .ok());
  EXPECT_EQ(device.releases, 1);
  EXPECT_EQ(pinned.releases, 1);

  device.device = 0;
  auto arenas = QwenNumericalTapArenas::AllocateVerified(
                    plan, device, pinned, placement, 0, 2)
                    .value();
  EXPECT_FALSE(arenas.device_endpoint(2, 1, 0).ok());
  EXPECT_FALSE(arenas.pinned_endpoint(0, 0, 0).ok());
}

}  // namespace
}  // namespace pih
