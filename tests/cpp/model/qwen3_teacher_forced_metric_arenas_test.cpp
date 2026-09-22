#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_metric_arenas.h"

namespace pih {
namespace {

static_assert(std::is_move_constructible_v<QwenTeacherForcedMetricArenas>);
static_assert(!std::is_move_assignable_v<QwenTeacherForcedMetricArenas>);

class DeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    if (calls == fail_at) return Status::ResourceExhausted("injected device");
    return Allocation{reinterpret_cast<void*>(
        UINT64_C(0x100000000) + UINT64_C(0x100000000) * calls),
        bytes, alignment, static_cast<std::uint64_t>(calls),
        Device::Create(DeviceType::kCuda, device).value()};
  }
  void deallocate(Allocation) noexcept override { ++releases; }
  int calls = 0, releases = 0, fail_at = 99, device = 0;
};

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    if (calls == fail_at) return Status::ResourceExhausted("injected pinned");
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment,
                      static_cast<std::uint64_t>(100 + calls), Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    ++releases;
    ::operator delete(value.data, std::align_val_t(value.alignment));
  }
  int calls = 0, releases = 0, fail_at = 99;
};

class Placement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void* data, std::uint64_t bytes,
                std::int32_t numa) override {
    ++calls;
    if (calls == fail_at || data == nullptr || bytes == 0 || numa != 2)
      return Status::FailedPrecondition("injected placement");
    return Status::Ok();
  }
  int calls = 0, fail_at = 99;
};

TEST(QwenTeacherForcedMetricArenasTest, OwnsTypedDeviceAndPinnedRanges) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  {
    auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
        17, device, pinned, placement, 0, 2);
    ASSERT_TRUE(arenas.ok()) << arenas.status().message();
    EXPECT_EQ(arenas->pinned_sample_rows().size(),
              17U * sizeof(std::uint32_t));
    EXPECT_EQ(arenas->pinned_targets().size(), 17U * sizeof(std::uint32_t));
    EXPECT_EQ(arenas->pinned_result().size(), arenas->layout().total_bytes());
    EXPECT_EQ(arenas->device_targets(17)->dtype(), DType::kUInt32);
    EXPECT_EQ(arenas->device_argmax(17)->dim(0), 17);
    EXPECT_EQ(arenas->device_nll(17)->dtype(), DType::kFloat64);
    EXPECT_EQ(arenas->device_nonfinite(17)->dtype(), DType::kUInt32);
    EXPECT_EQ(arenas->device_error()->dim(0), 1);
    EXPECT_EQ(arenas->pinned_sample_rows_endpoint(10)->device_or_numa, 2);
    EXPECT_EQ(arenas->device_sample_rows_endpoint(10)->memory_type,
              CudaCopyMemoryType::kDevice);
    EXPECT_EQ(arenas->pinned_targets_endpoint(11)->device_or_numa, 2);
    EXPECT_EQ(arenas->device_result_endpoint(12)->memory_type,
              CudaCopyMemoryType::kDevice);
    EXPECT_FALSE(arenas->device_targets(18).ok());
    EXPECT_FALSE(arenas->pinned_result_endpoint(0).ok());
  }
  EXPECT_EQ(device.releases, 3);
  EXPECT_EQ(pinned.releases, 3);
  EXPECT_EQ(placement.calls, 3);
}

TEST(QwenTeacherForcedMetricArenasTest, RollsBackAllocationAndPlacementFailure) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  pinned.fail_at = 3;
  EXPECT_FALSE(QwenTeacherForcedMetricArenas::AllocateVerified(
                   4, device, pinned, placement, 0, 2).ok());
  EXPECT_EQ(device.releases, 3);
  EXPECT_EQ(pinned.releases, 2);

  DeviceAllocator device2;
  PinnedAllocator pinned2;
  Placement placement2;
  placement2.fail_at = 3;
  EXPECT_FALSE(QwenTeacherForcedMetricArenas::AllocateVerified(
                   4, device2, pinned2, placement2, 0, 2).ok());
  EXPECT_EQ(device2.releases, 3);
  EXPECT_EQ(pinned2.releases, 3);
}

TEST(QwenTeacherForcedMetricArenasTest, RejectsWrongDeviceAndPlacementIdentity) {
  DeviceAllocator device;
  device.device = 1;
  PinnedAllocator pinned;
  Placement placement;
  EXPECT_FALSE(QwenTeacherForcedMetricArenas::AllocateVerified(
                   1, device, pinned, placement, 0, 2).ok());
  EXPECT_FALSE(QwenTeacherForcedMetricArenas::AllocateVerified(
                   1, device, pinned, placement, -1, 2).ok());
}

TEST(QwenTeacherForcedMetricArenasTest,
     PreparedLeaseDestructionAllowsAnotherTransaction) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, device, pinned, placement, 0, 2).value();
  {
    auto lease = arenas.acquire_transaction_lease();
    ASSERT_TRUE(lease.ok());
    EXPECT_FALSE(arenas.acquire_transaction_lease().ok());
  }
  EXPECT_TRUE(arenas.acquire_transaction_lease().ok());
}

TEST(QwenTeacherForcedMetricArenasTest,
     AssignsDistinctProcessArenaIdentities) {
  DeviceAllocator first_device;
  PinnedAllocator first_pinned;
  Placement first_placement;
  auto first = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, first_device, first_pinned, first_placement, 0, 2).value();
  DeviceAllocator second_device;
  PinnedAllocator second_pinned;
  Placement second_placement;
  auto second = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, second_device, second_pinned, second_placement, 0, 2).value();
  EXPECT_NE(first.arena_identity(), 0U);
  EXPECT_NE(second.arena_identity(), 0U);
  EXPECT_NE(first.arena_identity(), second.arena_identity());
  EXPECT_EQ(first.reserve_factory_generation().value(), 1U);
  EXPECT_EQ(second.reserve_factory_generation().value(), 1U);
}

TEST(QwenTeacherForcedMetricArenasTest,
     AuthorizedCompletionReleasesSubmittedLease) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, device, pinned, placement, 0, 2).value();
  auto lease = arenas.acquire_transaction_lease().value();
  ASSERT_TRUE(lease.mark_submitted().ok());
  EXPECT_FALSE(arenas.acquire_transaction_lease().ok());
  ASSERT_TRUE(lease.release_completed().ok());
  EXPECT_TRUE(arenas.acquire_transaction_lease().ok());
}

TEST(QwenTeacherForcedMetricArenasTest,
     AbandonedSubmittedLeaseFailStopsArenaReuse) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, device, pinned, placement, 0, 2).value();
  {
    auto lease = arenas.acquire_transaction_lease().value();
    ASSERT_TRUE(lease.mark_submitted().ok());
  }
  EXPECT_FALSE(arenas.acquire_transaction_lease().ok());
}

}  // namespace
}  // namespace pih
