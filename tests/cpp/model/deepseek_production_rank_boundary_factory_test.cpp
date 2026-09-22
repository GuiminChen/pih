#include "pih/model/deepseek_production_rank_boundary_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class FactoryAllocator final : public Allocator {
 public:
  explicit FactoryAllocator(std::int32_t rank)
      : device_(Device::Create(DeviceType::kCuda, rank).value()) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
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

class FactoryRuntime final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,
                                                std::uint32_t) override {
    return 77;
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override { return 88; }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t) override { return next_event_++; }
  void destroy_event(DriverEventHandle) noexcept override {}
  void destroy_stream(DriverStreamHandle) noexcept override {}
  void release_primary_context(std::int32_t,
                               std::uintptr_t) noexcept override {}
 private:
  DriverEventHandle next_event_ = 100;
};

class FactoryTransport final : public DeepSeekBoundaryTransportDriver {
 public:
  Status bind_p2p(DeepSeekNcclRole, void*, std::uint64_t,
                  std::uintptr_t) override { return Status::Ok(); }
  Status group_start() override { return Status::Ok(); }
  Status send(std::uint64_t, std::uint32_t) override { return Status::Ok(); }
  Status recv(std::uint64_t, std::uint32_t) override { return Status::Ok(); }
  Result<DeepSeekNcclAsyncStatus> group_end() override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  Result<DeepSeekNcclAsyncStatus> async_status() override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
};

class FactoryEvent final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
};

class FactoryEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class FactoryDriverOwner final : public DeepSeekBoundaryDriverOwner {
 public:
  explicit FactoryDriverOwner(DeepSeekBoundaryTransportDriver& transport)
      : transport_(&transport) {}
  DeepSeekNcclP2pBindingTarget& binding() noexcept override {
    return *transport_;
  }
  DeepSeekNcclP2pDriver& nccl() noexcept override { return *transport_; }
  CompletionEventDriver& event() noexcept override { return event_; }
  CompletionEvidenceProvider& evidence() noexcept override {
    return evidence_;
  }
 private:
  DeepSeekBoundaryTransportDriver* transport_;
  FactoryEvent event_;
  FactoryEvidence evidence_;
};

class FactoryDriverFactory final : public DeepSeekBoundaryDriverOwnerFactory {
 public:
  Result<std::unique_ptr<DeepSeekBoundaryDriverOwner>> create(
      DeepSeekBoundaryTransportDriver& transport) override {
    ++calls;
    return std::unique_ptr<DeepSeekBoundaryDriverOwner>(
        std::make_unique<FactoryDriverOwner>(transport));
  }
  int calls = 0;
};

class FactoryClock final : public DeepSeekBoundaryPreparationClock {
 public:
  Result<std::uint64_t> now_ns() override { return 100; }
};

TEST(DeepSeekProductionRankBoundaryFactoryTest,
     MiddleRankPreparesBothBoundariesAndCanAbortBeforeCommit) {
  FactoryAllocator allocator(1);
  FactoryRuntime runtime;
  auto device_resources = DeepSeekRankBoundaryDeviceResources::Allocate(
      1, 3, 2, 77, allocator, runtime).value();
  FactoryTransport incoming_transport;
  FactoryTransport outgoing_transport;
  FactoryClock clock;
  FactoryDriverFactory drivers;
  auto factory = DeepSeekProductionRankBoundaryFactory::Create(
      {2, 1, 3, 77, 88, 9, 1000, {31, 32}},
      &incoming_transport, &outgoing_transport,
      std::move(device_resources), clock, drivers);
  ASSERT_TRUE(factory.ok()) << factory.status().message();

  auto capacity = DeepSeekPipelineCapacity::Create(3, 8, 8, 1, false).value();
  auto pipeline = DeepSeekPipelineResourceSet::Create(capacity).value();
  auto transaction = pipeline.prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2}).value();
  auto output = Buffer::Allocate(allocator, 65536, 256).value();
  auto source = DeepSeekBoundarySendSource::Create(
      output, 0, 65536, 2, 41, 77, 7).value();
  const DeepSeekStagePlan stage{1, {21, 40}, false, false, false};
  {
    auto prepared = (*factory)->prepare(transaction, stage, 3, 2, source);
    ASSERT_TRUE(prepared.ok()) << prepared.status().message();
    EXPECT_NE(prepared->incoming, nullptr);
    EXPECT_NE(prepared->outgoing, nullptr);
    EXPECT_NE(prepared->incoming->leased_buffer_address(), 0U);
    EXPECT_EQ(prepared->incoming->leased_buffer_bytes(), 65536U);
    EXPECT_EQ(prepared->outgoing->leased_buffer_address(),
              reinterpret_cast<std::uintptr_t>(output.data()));
    EXPECT_EQ(prepared->outgoing->leased_buffer_bytes(), 65536U);
    EXPECT_NE(prepared->drivers.incoming.owner, nullptr);
    EXPECT_NE(prepared->drivers.outgoing.owner, nullptr);
    EXPECT_EQ(drivers.calls, 2);
  }
  auto prepared_again = (*factory)->prepare(
      transaction, stage, 3, 2, source);
  ASSERT_TRUE(prepared_again.ok()) << prepared_again.status().message();
  EXPECT_EQ(drivers.calls, 4);
}

TEST(DeepSeekProductionRankBoundaryFactoryTest,
     RejectsTopologyAndOutgoingSourceMismatch) {
  FactoryAllocator allocator(0);
  FactoryRuntime runtime;
  FactoryTransport outgoing_transport;
  FactoryClock clock;
  FactoryDriverFactory drivers;
  auto resources = DeepSeekRankBoundaryDeviceResources::Allocate(
      0, 2, 2, 77, allocator, runtime).value();
  auto factory = DeepSeekProductionRankBoundaryFactory::Create(
      {2, 0, 2, 77, 88, 9, 1000, {0, 0}}, nullptr,
      &outgoing_transport, std::move(resources), clock, drivers);
  ASSERT_TRUE(factory.ok());
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false).value();
  auto pipeline = DeepSeekPipelineResourceSet::Create(capacity).value();
  auto transaction = pipeline.prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2}).value();
  const DeepSeekStagePlan stage{0, {0, 30}, true, false, false};
  EXPECT_FALSE((*factory)->prepare(transaction, stage, 2, 2, {}).ok());
}

TEST(DeepSeekProductionRankBoundaryFactoryTest,
     SteadyClockPublishesPositiveMonotonicTime) {
  DeepSeekSteadyBoundaryClock clock;
  auto first = clock.now_ns();
  auto second = clock.now_ns();
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_GT(*first, 0U);
  EXPECT_GE(*second, *first);
}

}  // namespace
}  // namespace pih
