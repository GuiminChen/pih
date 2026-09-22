#include "pih/model/deepseek_boundary_operation_builder.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class OperationAllocator final : public Allocator {
 public:
  explicit OperationAllocator(Device device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    return Allocation{data, bytes, alignment, generation_++, device_};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  Device device_;
  std::uint64_t generation_ = 17;
};

class OperationResourceDriver final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,
                                                std::uint32_t) override {
    return 77;
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override { return 1; }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t) override { return next_event_++; }
  void destroy_event(DriverEventHandle) noexcept override {}
  void destroy_stream(DriverStreamHandle) noexcept override {}
  void release_primary_context(std::int32_t,
                               std::uintptr_t) noexcept override {}
 private:
  DriverEventHandle next_event_ = 100;
};

TEST(DeepSeekBoundaryOperationBuilderTest,
     ReservesTrackedCreditAndRollsBackPreparationFailure) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false).value();
  auto resources = DeepSeekPipelineResourceSet::Create(capacity).value();
  auto transaction = resources.prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2}).value();
  auto sequencer = DeepSeekNcclOperationSequencer::Create(1).value();
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  OperationAllocator allocator(
      Device::Create(DeviceType::kCuda, 0).value());
  OperationResourceDriver runtime;
  auto boundary_resources = DeepSeekRankBoundaryDeviceResources::Allocate(
      0, 2, 2, 77, allocator, runtime).value();
  auto buffer = Buffer::Allocate(allocator, 65536, 256).value();
  auto source = DeepSeekBoundarySendSource::Create(
      buffer, 0, 65536, 2, 9, 77, 3).value();

  auto operation = DeepSeekBoundaryOperationBuilder::CreateSend(
      transaction, 2, 0, 4, 2, source, *boundary_resources, 88,
      sequencer, tracker,
      100, 200);
  ASSERT_TRUE(operation.ok()) << operation.status().message();
  EXPECT_EQ(tracker.state(0), DeepSeekBoundaryCreditState::kPrepared);
  operation->reset();
  EXPECT_EQ(tracker.state(0), DeepSeekBoundaryCreditState::kFree);

  auto failed = DeepSeekBoundaryOperationBuilder::CreateSend(
      transaction, 2, 0, 4, 2, source, *boundary_resources, 0,
      sequencer, tracker,
      100, 200);
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(tracker.state(0), DeepSeekBoundaryCreditState::kFree);
  EXPECT_FALSE(tracker.poisoned());
}

TEST(DeepSeekBoundaryOperationBuilderTest,
     ReceiveBufferAlwaysMatchesReservedCreditIndex) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false).value();
  auto pipeline = DeepSeekPipelineResourceSet::Create(capacity).value();
  auto transaction = pipeline.prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2}).value();
  auto sequencer = DeepSeekNcclOperationSequencer::Create(1).value();
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  auto occupied = tracker.reserve(99, 99).value();
  ASSERT_EQ(occupied.credit_index, 0U);
  OperationAllocator allocator(
      Device::Create(DeviceType::kCuda, 1).value());
  OperationResourceDriver runtime;
  auto resources = DeepSeekRankBoundaryDeviceResources::Allocate(
      1, 2, 2, 77, allocator, runtime).value();

  auto operation = DeepSeekBoundaryOperationBuilder::CreateRecv(
      transaction, 2, 0, 4, 2, *resources, {31, 32}, 77, 88,
      sequencer, tracker, 100, 200);
  ASSERT_TRUE(operation.ok()) << operation.status().message();
  EXPECT_EQ((*operation)->manifest().buffer_owner_id, 32U);
  ASSERT_NE(resources->incoming_slot(1), nullptr);
  EXPECT_EQ((*operation)->manifest().buffer_generation,
            resources->incoming_slot(1)->generation());
  EXPECT_EQ(tracker.state(1), DeepSeekBoundaryCreditState::kPrepared);
  operation->reset();
  EXPECT_EQ(tracker.state(1), DeepSeekBoundaryCreditState::kFree);
  EXPECT_TRUE(tracker.abort_prepare(occupied).ok());
}

}  // namespace
}  // namespace pih
