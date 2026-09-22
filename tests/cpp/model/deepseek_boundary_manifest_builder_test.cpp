#include "pih/model/deepseek_boundary_manifest_builder.h"

#include <gtest/gtest.h>

#include <cstdlib>

namespace pih {
namespace {

class ManifestAllocator final : public Allocator {
 public:
  explicit ManifestAllocator(Device device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("allocation failed");
    return Allocation{data, bytes, alignment, generation_++, device_};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }

 private:
  Device device_;
  std::uint64_t generation_ = 1;
};

TEST(DeepSeekBoundaryManifestBuilderTest,
     BuildsPairWithSharedGlobalAndProjectedLocalOrdinals) {
  ManifestAllocator send_allocator(
      Device::Create(DeviceType::kCuda, 1).value());
  ManifestAllocator recv_allocator(
      Device::Create(DeviceType::kCuda, 2).value());
  auto send_buffer = Buffer::Allocate(send_allocator, 65536, 256).value();
  auto recv_buffer = Buffer::Allocate(recv_allocator, 131072, 256).value();
  auto source = DeepSeekBoundarySendSource::Create(
      send_buffer, 0, 65536, 2, 41, 71, 9).value();
  const DeepSeekPipelinePlanDescriptor descriptor{
      .engine_epoch = 3, .plan_sequence = 5,
      .phase = DeepSeekPlanPhase::kDecode, .token_count = 2,
      .sequence_count = 2};

  auto send = DeepSeekBoundaryManifestBuilder::CreateSend(
      descriptor, 3, 1, 12, 2, source);
  auto recv = DeepSeekBoundaryManifestBuilder::CreateRecv(
      descriptor, 3, 1, 12, 2, recv_buffer, 42, 71);

  ASSERT_TRUE(send.ok()) << send.status().message();
  ASSERT_TRUE(recv.ok()) << recv.status().message();
  EXPECT_EQ(send->operation_plan_id, send->global_issue_ordinal);
  EXPECT_EQ(send->operation_plan_id, recv->operation_plan_id);
  EXPECT_EQ(send->global_issue_ordinal, recv->global_issue_ordinal);
  EXPECT_EQ(send->operation_ordinal, 10U);
  EXPECT_EQ(recv->operation_ordinal, 5U);
  EXPECT_EQ(send->buffer_owner_id, 41U);
  EXPECT_EQ(send->buffer_capacity_bytes, 65536U);
  EXPECT_EQ(recv->buffer_owner_id, 42U);
  EXPECT_EQ(recv->buffer_capacity_bytes, 131072U);
  EXPECT_TRUE(DeepSeekNcclP2pPlan::ValidatePair(*send, *recv).ok());
}

TEST(DeepSeekBoundaryManifestBuilderTest,
     RejectsWireSourceAndReceiveCapacityMismatch) {
  ManifestAllocator send_allocator(
      Device::Create(DeviceType::kCuda, 0).value());
  ManifestAllocator recv_allocator(
      Device::Create(DeviceType::kCuda, 1).value());
  auto send_buffer = Buffer::Allocate(send_allocator, 65536, 256).value();
  auto short_recv = Buffer::Allocate(recv_allocator, 32768, 256).value();
  auto source = DeepSeekBoundarySendSource::Create(
      send_buffer, 0, 65536, 2, 41, 71, 9).value();
  const DeepSeekPipelinePlanDescriptor descriptor{
      .engine_epoch = 3, .plan_sequence = 5,
      .phase = DeepSeekPlanPhase::kDrain, .token_count = 0,
      .sequence_count = 0};

  EXPECT_FALSE(DeepSeekBoundaryManifestBuilder::CreateSend(
      descriptor, 2, 0, 12, 1, source).ok());
  EXPECT_FALSE(DeepSeekBoundaryManifestBuilder::CreateRecv(
      descriptor, 2, 0, 12, 2, short_recv, 42, 71).ok());
  EXPECT_FALSE(DeepSeekBoundaryManifestBuilder::CreateRecv(
      descriptor, 2, 0, 0, 1, short_recv, 42, 71).ok());
}

}  // namespace
}  // namespace pih
