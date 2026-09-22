#include "pih/model/deepseek_nccl_boundary_lease.h"

#include <gtest/gtest.h>

#include <new>

namespace pih {
namespace {

class BoundaryAllocator final : public Allocator {
 public:
  explicit BoundaryAllocator(Device device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t{static_cast<std::size_t>(alignment)});
    return Allocation{data, bytes, alignment, 17, device_};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data,
                      std::align_val_t{static_cast<std::size_t>(allocation.alignment)});
  }
 private:
  Device device_;
};

class BindingTarget final : public DeepSeekNcclP2pBindingTarget {
 public:
  Status bind_p2p(DeepSeekNcclRole role, void* buffer,
                  std::uint64_t bytes, std::uintptr_t stream) override {
    observed_role = role; observed_buffer = buffer;
    observed_bytes = bytes; observed_stream = stream; return Status::Ok();
  }
  DeepSeekNcclRole observed_role = DeepSeekNcclRole::kRecv;
  void* observed_buffer = nullptr;
  std::uint64_t observed_bytes = 0;
  std::uintptr_t observed_stream = 0;
};

DeepSeekNcclP2pManifest boundary_manifest() {
  return {.operation_plan_id = 1,
          .engine_epoch = 2,
          .communicator_generation = 3,
          .pipeline_plan_sequence = 4,
          .operation_ordinal = 5,
          .global_issue_ordinal = 5,
          .directed_boundary_id = 0,
          .role = DeepSeekNcclRole::kSend,
          .local_global_rank = 0,
          .peer_global_rank = 1,
          .communicator_local_rank = 0,
          .communicator_peer_rank = 1,
          .buffer_owner_id = 9,
          .buffer_offset_bytes = 256,
          .buffer_capacity_bytes = 131072,
          .buffer_generation = 17,
          .context_identity = 77,
          .token_count = 2};
}

TEST(DeepSeekNcclBoundaryLeaseTest, BindsExactGenerationSpanAndStream) {
  auto cuda = Device::Create(DeviceType::kCuda, 0);
  ASSERT_TRUE(cuda.ok());
  BoundaryAllocator allocator(*cuda);
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  ASSERT_TRUE(buffer.ok());
  auto lease = DeepSeekNcclBoundaryLease::Create(
      boundary_manifest(), *buffer, 9, 77, 88);
  ASSERT_TRUE(lease.ok()) << lease.status().message();
  EXPECT_EQ(lease->data(), static_cast<std::byte*>(buffer->data()) + 256);
  EXPECT_EQ(lease->bytes(), 65536U);
  BindingTarget target;
  ASSERT_TRUE(lease->bind(target).ok());
  EXPECT_EQ(target.observed_role, DeepSeekNcclRole::kSend);
  EXPECT_EQ(target.observed_buffer, lease->data());
  EXPECT_EQ(target.observed_bytes, 65536U);
  EXPECT_EQ(target.observed_stream, 88U);
}

TEST(DeepSeekNcclBoundaryLeaseTest, RejectsStaleOwnerGenerationAndCpuBuffer) {
  auto cuda = Device::Create(DeviceType::kCuda, 0);
  ASSERT_TRUE(cuda.ok());
  BoundaryAllocator allocator(*cuda);
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  ASSERT_TRUE(buffer.ok());
  auto manifest = boundary_manifest();
  EXPECT_FALSE(DeepSeekNcclBoundaryLease::Create(
      manifest, *buffer, 10, 77, 88).ok());
  manifest.buffer_generation = 18;
  EXPECT_FALSE(DeepSeekNcclBoundaryLease::Create(
      manifest, *buffer, 9, 77, 88).ok());
  manifest = boundary_manifest();
  EXPECT_FALSE(DeepSeekNcclBoundaryLease::Create(
      manifest, *buffer, 9, 78, 88).ok());

  BoundaryAllocator cpu_allocator(Device::Cpu());
  auto cpu = Buffer::Allocate(cpu_allocator, 131072, 256);
  ASSERT_TRUE(cpu.ok());
  EXPECT_FALSE(DeepSeekNcclBoundaryLease::Create(
      boundary_manifest(), *cpu, 9, 77, 88).ok());
}

}  // namespace
}  // namespace pih
