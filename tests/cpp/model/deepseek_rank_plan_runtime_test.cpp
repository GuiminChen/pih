#include "pih/model/deepseek_rank_plan_runtime.h"
#include "pih/model/deepseek_rank_serving_plan_executor.h"

#include <gtest/gtest.h>

#include <memory>

namespace pih {
namespace {

class BoundaryDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("boundary buffer");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class CountingProvider final : public DeepSeekRankPlanResourceProvider {
 public:
  class Lease final : public DeepSeekRankPlanResourceLease {
   public:
    explicit Lease(std::size_t& active) : active_(&active) { ++*active_; }
    ~Lease() override { --*active_; }
   private:
    std::size_t* active_;
  };

  Result<std::unique_ptr<DeepSeekRankPlanResourceLease>> reserve(
      const DeepSeekPipelinePlanDescriptor&, const DeepSeekStagePlan&,
      DeepSeekRankPlanResourceRequirement) override {
    return std::unique_ptr<DeepSeekRankPlanResourceLease>(
        std::make_unique<Lease>(active));
  }

  std::size_t active = 0;
};

class ScriptedCompute final : public DeepSeekStageComputeDriver {
 public:
  Status launch(const DeepSeekPipelinePlanDescriptor&,
                const DeepSeekStagePlan&) override {
    ++launches;
    return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    ++polls;
    return poll_result;
  }

  std::size_t launches = 0;
  std::size_t polls = 0;
  DeepSeekStageComputeStatus poll_result =
      DeepSeekStageComputeStatus::kSuccess;
};

class CombinedBoundaryOwner final : public DeepSeekBoundaryDriverOwner,
                                    public DeepSeekNcclP2pBindingTarget,
                                    public DeepSeekNcclP2pDriver,
                                    public CompletionEventDriver,
                                    public CompletionEvidenceProvider {
 public:
  DeepSeekNcclP2pBindingTarget& binding() noexcept override { return *this; }
  DeepSeekNcclP2pDriver& nccl() noexcept override { return *this; }
  CompletionEventDriver& event() noexcept override { return *this; }
  CompletionEvidenceProvider& evidence() noexcept override { return *this; }
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
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class CompositeTransport final : public DeepSeekBoundaryTransportDriver {
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

class CompositeEvent final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
};

class CompositeEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class PreparedBoundaryOperation final : public DeepSeekBoundaryOperation {
 public:
  explicit PreparedBoundaryOperation(
      std::uint64_t bytes = 32768,
      std::uintptr_t address = 0xABC000)
      : bytes_(bytes), address_(address) {}
  Status submit(DeepSeekNcclP2pBindingTarget&, DeepSeekNcclP2pDriver&,
                CompletionEventDriver&) override {
    state_ = DeepSeekBoundaryExecutorState::kCompleteVerified;
    return Status::Ok();
  }
  Status poll_issue(DeepSeekNcclP2pDriver&,
                    CompletionEventDriver&) override {
    return Status::FailedPrecondition("not pending");
  }
  Status poll_completion(DeepSeekNcclP2pDriver&, CompletionEventDriver&,
                         CompletionEvidenceProvider&) override {
    return Status::FailedPrecondition("not in flight");
  }
  DeepSeekBoundaryExecutorState state() const noexcept override {
    return state_;
  }
  std::uintptr_t leased_buffer_address() const noexcept override {
    return address_;
  }
  std::uint64_t leased_buffer_bytes() const noexcept override {
    return bytes_;
  }
 private:
  std::uint64_t bytes_ = 0;
  std::uintptr_t address_ = 0;
  DeepSeekBoundaryExecutorState state_ = DeepSeekBoundaryExecutorState::kReady;
};

class PreparedBoundaryFactory final : public DeepSeekRankBoundaryFactory {
 public:
  Result<DeepSeekPreparedRankBoundaries> prepare(
      DeepSeekPipelineTransaction& transaction, const DeepSeekStagePlan& stage,
      std::uint32_t world_size,
      std::uint32_t,
      const std::optional<DeepSeekBoundarySendSource>& outgoing_source)
      override {
    ++calls;
    observed_transaction_state = transaction.state();
    DeepSeekPreparedRankBoundaries prepared;
    if (stage.rank > 0) {
      prepared.incoming =
          std::make_unique<PreparedBoundaryOperation>(incoming_bytes);
      prepared.drivers.incoming.owner =
          std::make_unique<CombinedBoundaryOwner>();
    }
    if (stage.rank + 1 < world_size) {
      prepared.outgoing = std::make_unique<PreparedBoundaryOperation>(
          outgoing_source.has_value() ? outgoing_source->wire_bytes()
                                      : 32768,
          outgoing_source.has_value()
              ? reinterpret_cast<std::uintptr_t>(outgoing_source->data())
              : 0xABC000);
      prepared.drivers.outgoing.owner =
          std::make_unique<CombinedBoundaryOwner>();
    }
    return prepared;
  }
  int calls = 0;
  std::uint64_t incoming_bytes = 32768;
  DeepSeekPipelineTransactionState observed_transaction_state =
      DeepSeekPipelineTransactionState::kCommitted;
};

TEST(DeepSeekRankPlanRuntimeTest,
     CompositeOwnerBorrowsOneTransportForBindingAndNccl) {
  auto transport = std::make_unique<CompositeTransport>();
  auto event = std::make_unique<CompositeEvent>();
  auto evidence = std::make_unique<CompositeEvidence>();
  auto* transport_identity = transport.get();
  auto* event_identity = event.get();
  auto* evidence_identity = evidence.get();

  auto owner = DeepSeekCompositeBoundaryDriverOwner::Create(
      std::move(transport), std::move(event), std::move(evidence));
  ASSERT_TRUE(owner.ok()) << owner.status().message();
  EXPECT_EQ(&owner->binding(), transport_identity);
  EXPECT_EQ(&owner->nccl(), transport_identity);
  EXPECT_EQ(&owner->event(), event_identity);
  EXPECT_EQ(&owner->evidence(), evidence_identity);
}

TEST(DeepSeekRankPlanRuntimeTest, CompositeOwnerRejectsMissingComponent) {
  auto result = DeepSeekCompositeBoundaryDriverOwner::Create(
      std::make_unique<CompositeTransport>(),
      std::unique_ptr<CompletionEventDriver>{},
      std::make_unique<CompositeEvidence>());
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(DeepSeekRankPlanRuntimeTest,
     BorrowedOwnerDoesNotTakeGenerationTransportOwnership) {
  CompositeTransport transport;
  auto event = std::make_unique<CompositeEvent>();
  auto evidence = std::make_unique<CompositeEvidence>();
  auto* event_identity = event.get();
  auto* evidence_identity = evidence.get();

  auto owner = DeepSeekBorrowedBoundaryDriverOwner::Create(
      transport, std::move(event), std::move(evidence));
  ASSERT_TRUE(owner.ok()) << owner.status().message();
  EXPECT_EQ(&owner->binding(), &transport);
  EXPECT_EQ(&owner->nccl(), &transport);
  EXPECT_EQ(&owner->event(), event_identity);
  EXPECT_EQ(&owner->evidence(), evidence_identity);
}

TEST(DeepSeekRankPlanRuntimeTest, BorrowedOwnerRejectsMissingPlanEvidence) {
  CompositeTransport transport;
  auto result = DeepSeekBorrowedBoundaryDriverOwner::Create(
      transport, std::make_unique<CompositeEvent>(),
      std::unique_ptr<CompletionEvidenceProvider>{});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(DeepSeekRankPlanRuntimeTest,
     BoundaryFactoryBindsAgainstStablePreparedTransaction) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  ASSERT_TRUE(capacity.ok());
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(resources.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  auto transaction = resources->prepare(descriptor);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  CountingProvider provider;
  auto reservation = DeepSeekRankPlanReservation::Prepare(
      descriptor, plan->rank(0), 2,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 128, 256, provider);
  ASSERT_TRUE(reservation.ok());
  ScriptedCompute compute;
  PreparedBoundaryFactory factory;
  BoundaryDeviceAllocator boundary_allocator;
  auto boundary_buffer = Buffer::Allocate(
      boundary_allocator, DeepSeekBoundarySendSource::kWireBytesPerToken,
      256).value();
  auto outgoing = DeepSeekBoundarySendSource::Create(
      boundary_buffer, 0, DeepSeekBoundarySendSource::kWireBytesPerToken,
      1, 41, 77, 1).value();

  auto runtime = DeepSeekRankPlanRuntime::CreateBorrowedComputeWithBoundaries(
      std::move(*transaction), plan->rank(0), 2, std::move(*reservation),
      compute, factory, 1, outgoing);
  ASSERT_TRUE(runtime.ok()) << runtime.status().message();
  EXPECT_EQ(factory.calls, 1);
  EXPECT_EQ(factory.observed_transaction_state,
            DeepSeekPipelineTransactionState::kPrepared);
  ASSERT_TRUE(runtime->mark_ready().ok());
  ASSERT_TRUE(runtime->commit().ok());
  ASSERT_TRUE(runtime->advance().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kExecuting);
  ASSERT_TRUE(runtime->advance().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kComplete);
}

TEST(DeepSeekRankPlanRuntimeTest,
     PreparedBoundaryOwnershipExposesIncomingBeforeRuntimeFinalization) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  ASSERT_TRUE(capacity.ok()) << capacity.status().message();
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  const DeepSeekPipelinePlanDescriptor descriptor{
      77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  auto transaction = resources->prepare(descriptor);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok()) << transaction.status().message();
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  PreparedBoundaryFactory factory;
  auto prepared = DeepSeekPreparedRankRuntimeBoundaries::Prepare(
      std::move(*transaction), plan->rank(1), 2, factory, 1,
      std::nullopt);
  ASSERT_TRUE(prepared.ok()) << prepared.status().message();
  EXPECT_EQ(prepared->incoming_activation_bf16(), 0xABC000U);
  EXPECT_EQ(prepared->incoming_capacity_bytes(), 32768U);

  CountingProvider provider;
  auto reservation = DeepSeekRankPlanReservation::Prepare(
      descriptor, plan->rank(1), 2,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 128, 256, provider);
  ASSERT_TRUE(reservation.ok()) << reservation.status().message();
  ScriptedCompute compute;
  auto runtime =
      DeepSeekRankPlanRuntime::CreateBorrowedComputeWithPreparedBoundaries(
          std::move(*prepared), std::move(*reservation), compute);
  ASSERT_TRUE(runtime.ok()) << runtime.status().message();
  ASSERT_TRUE(runtime->mark_ready().ok());
  ASSERT_TRUE(runtime->commit().ok());
}

TEST(DeepSeekRankPlanRuntimeTest,
     PreparedBoundaryRejectsIncomingLeaseSmallerThanWirePayload) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  const DeepSeekPipelinePlanDescriptor descriptor{
      77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  auto transaction = resources->prepare(descriptor);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(plan.ok());
  PreparedBoundaryFactory factory;
  factory.incoming_bytes = 32768;
  EXPECT_FALSE(DeepSeekPreparedRankRuntimeBoundaries::Prepare(
                   std::move(*transaction), plan->rank(1), 2, factory, 2,
                   std::nullopt)
                   .ok());
}

TEST(DeepSeekRankPlanRuntimeTest,
     PreparedOutgoingBoundaryRequiresExactSendSourceIdentity) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  const DeepSeekPipelinePlanDescriptor descriptor{
      77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  auto transaction = resources->prepare(descriptor);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(plan.ok());
  PreparedBoundaryFactory factory;
  EXPECT_FALSE(DeepSeekPreparedRankRuntimeBoundaries::Prepare(
                   std::move(*transaction), plan->rank(0), 2, factory, 1,
                   std::nullopt)
                   .ok());
}

TEST(DeepSeekRankPlanRuntimeTest,
     PpTwoThroughFourPreparedBoundaryRollbackReleasesEveryRankTransaction) {
  for (std::uint32_t world_size = 2; world_size <= 4; ++world_size) {
    auto capacity = DeepSeekPipelineCapacity::Create(
        world_size, 8, 8, 1, false).value();
    auto topology = DeepSeekPipelinePlan::Create(world_size, false).value();
    std::vector<DeepSeekPipelineResourceSet> resource_sets;
    std::vector<std::unique_ptr<DeepSeekPreparedRankRuntimeBoundaries>>
        prepared;
    std::vector<std::unique_ptr<PreparedBoundaryFactory>> factories;
    BoundaryDeviceAllocator boundary_allocator;
    std::vector<std::unique_ptr<Buffer>> boundary_buffers;
    resource_sets.reserve(world_size);
    prepared.reserve(world_size);
    factories.reserve(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      resource_sets.push_back(
          DeepSeekPipelineResourceSet::Create(capacity).value());
      factories.push_back(std::make_unique<PreparedBoundaryFactory>());
    }
    const DeepSeekPipelinePlanDescriptor failed_descriptor{
        77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      auto transaction = resource_sets[rank].prepare(failed_descriptor);
      ASSERT_TRUE(transaction.ok()) << "PP" << world_size << " rank " << rank;
      std::optional<DeepSeekBoundarySendSource> outgoing;
      if (rank + 1 < world_size) {
        auto buffer = Buffer::Allocate(
            boundary_allocator,
            DeepSeekBoundarySendSource::kWireBytesPerToken, 256).value();
        boundary_buffers.push_back(
            std::make_unique<Buffer>(std::move(buffer)));
        outgoing = DeepSeekBoundarySendSource::Create(
            *boundary_buffers.back(), 0,
            DeepSeekBoundarySendSource::kWireBytesPerToken, 1,
            100 + rank, 77, 200 + rank).value();
      }
      auto boundaries = DeepSeekPreparedRankRuntimeBoundaries::Prepare(
          std::move(*transaction), topology.rank(rank), world_size,
          *factories[rank], 1, outgoing);
      ASSERT_TRUE(boundaries.ok()) << boundaries.status().message();
      if (rank == 0) {
        EXPECT_EQ(boundaries->incoming_activation_bf16(), 0U);
      } else {
        EXPECT_EQ(boundaries->incoming_activation_bf16(), 0xABC000U);
      }
      prepared.push_back(
          std::make_unique<DeepSeekPreparedRankRuntimeBoundaries>(
              std::move(*boundaries)));
    }

    // Models rollback after a later rank compiler rejects its input.
    prepared.clear();
    const DeepSeekPipelinePlanDescriptor retry_descriptor{
        77, 2, DeepSeekPlanPhase::kDecode, 1, 1};
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      auto retry = resource_sets[rank].prepare(retry_descriptor);
      ASSERT_TRUE(retry.ok()) << "PP" << world_size << " rank " << rank
                              << ": " << retry.status().message();
    }
  }
}

TEST(DeepSeekRankPlanRuntimeTest, BoundaryFactoryMustMatchRankTopology) {
  class EmptyFactory final : public DeepSeekRankBoundaryFactory {
   public:
    Result<DeepSeekPreparedRankBoundaries> prepare(
        DeepSeekPipelineTransaction&, const DeepSeekStagePlan&,
        std::uint32_t,
        std::uint32_t,
        const std::optional<DeepSeekBoundarySendSource>&) override {
      return DeepSeekPreparedRankBoundaries{};
    }
  } factory;
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  const DeepSeekPipelinePlanDescriptor descriptor{
      77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  auto transaction = resources->prepare(descriptor);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  CountingProvider provider;
  auto reservation = DeepSeekRankPlanReservation::Prepare(
      descriptor, plan->rank(0), 2,
      DeepSeekRoutedExpertResidency::kFullResident, 0, 128, 256, provider);
  ScriptedCompute compute;
  auto runtime = DeepSeekRankPlanRuntime::CreateBorrowedComputeWithBoundaries(
      std::move(*transaction), plan->rank(0), 2, std::move(*reservation),
      compute, factory, 1, std::nullopt);
  ASSERT_FALSE(runtime.ok());
  EXPECT_EQ(runtime.status().code(), StatusCode::kInvalidArgument);
}

struct RuntimeFixture final {
  DeepSeekPipelineResourceSet resources;
  CountingProvider provider;

  Result<DeepSeekRankPlanRuntime> create(ScriptedCompute*& compute) {
    auto capacity = DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false);
    if (!capacity.ok()) return capacity.status();
    auto resource_set = DeepSeekPipelineResourceSet::Create(*capacity);
    if (!resource_set.ok()) return resource_set.status();
    resources = std::move(*resource_set);
    const DeepSeekPipelinePlanDescriptor descriptor{
        77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
    auto transaction = resources.prepare(descriptor);
    if (!transaction.ok()) return transaction.status();
    auto plan = DeepSeekPipelinePlan::Create(1, false);
    if (!plan.ok()) return plan.status();
    auto reservation = DeepSeekRankPlanReservation::Prepare(
        descriptor, plan->rank(0), 1,
        DeepSeekRoutedExpertResidency::kFullResident, 0, 128, 256, provider);
    if (!reservation.ok()) return reservation.status();
    auto owned_compute = std::make_unique<ScriptedCompute>();
    compute = owned_compute.get();
    DeepSeekOwnedStageExecutionDrivers drivers;
    drivers.compute = std::move(owned_compute);
    return DeepSeekRankPlanRuntime::Create(
        std::move(*transaction), plan->rank(0), 1, std::move(*reservation),
        nullptr, nullptr, std::move(drivers));
  }

  Result<DeepSeekRankPlanRuntime> create_borrowed(
      ScriptedCompute& compute) {
    auto capacity = DeepSeekPipelineCapacity::Create(1, 8, 8, 1, false);
    if (!capacity.ok()) return capacity.status();
    auto resource_set = DeepSeekPipelineResourceSet::Create(*capacity);
    if (!resource_set.ok()) return resource_set.status();
    resources = std::move(*resource_set);
    const DeepSeekPipelinePlanDescriptor descriptor{
        77, 1, DeepSeekPlanPhase::kDecode, 1, 1};
    auto transaction = resources.prepare(descriptor);
    if (!transaction.ok()) return transaction.status();
    auto plan = DeepSeekPipelinePlan::Create(1, false);
    if (!plan.ok()) return plan.status();
    auto reservation = DeepSeekRankPlanReservation::Prepare(
        descriptor, plan->rank(0), 1,
        DeepSeekRoutedExpertResidency::kFullResident, 0, 128, 256, provider);
    if (!reservation.ok()) return reservation.status();
    return DeepSeekRankPlanRuntime::CreateBorrowedCompute(
        std::move(*transaction), plan->rank(0), 1, std::move(*reservation),
        nullptr, nullptr, compute, {});
  }
};

TEST(DeepSeekRankPlanRuntimeTest,
     RequiresPreparedReadyCommitAndRealExecutorCompletion) {
  RuntimeFixture fixture;
  ScriptedCompute* compute = nullptr;
  auto runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok()) << runtime.status().message();
  EXPECT_GT(fixture.provider.active, 0U);

  EXPECT_EQ(runtime->advance().code(), StatusCode::kFailedPrecondition);
  ASSERT_TRUE(runtime->mark_ready().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kReady);
  ASSERT_TRUE(runtime->commit().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kCommitted);
  EXPECT_GT(fixture.provider.active, 0U);

  ASSERT_TRUE(runtime->advance().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kExecuting);
  EXPECT_EQ(compute->launches, 1U);
  EXPECT_GT(fixture.provider.active, 0U);
  ASSERT_TRUE(runtime->advance().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kComplete);
  EXPECT_EQ(compute->polls, 1U);
  EXPECT_EQ(fixture.provider.active, 0U);
  auto next = fixture.resources.prepare(
      {77, 2, DeepSeekPlanPhase::kDecode, 1, 1});
  EXPECT_TRUE(next.ok()) << next.status().message();
}

TEST(DeepSeekRankPlanRuntimeTest,
     CancellationOnlyMarksCommittedWorkForDrain) {
  RuntimeFixture fixture;
  ScriptedCompute* compute = nullptr;
  auto runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok());
  EXPECT_EQ(runtime->cancel().code(), StatusCode::kFailedPrecondition);
  ASSERT_TRUE(runtime->mark_ready().ok());
  ASSERT_TRUE(runtime->commit().ok());
  ASSERT_TRUE(runtime->cancel().ok());
  EXPECT_TRUE(runtime->cancellation_requested());
  ASSERT_TRUE(runtime->advance().ok());
  ASSERT_TRUE(runtime->advance().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kComplete);
}

TEST(DeepSeekRankPlanRuntimeTest,
     ServingExecutionRequiresCommittedMatchingRuntimeAndDrainsCancellation) {
  RuntimeFixture fixture;
  ScriptedCompute* compute = nullptr;
  auto runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok());

  DeepSeekRankServingCommand command;
  command.session = {77, 8, 1, 0, 1, 2, 3,
                     Sha256Digest::ParseHex(
                         "0123456789abcdef0123456789abcdef"
                         "0123456789abcdef0123456789abcdef")
                         .value()};
  command.kind = DeepSeekRankServingCommandKind::kExecute;
  command.plan = runtime->descriptor();
  EXPECT_FALSE(DeepSeekRankPlanRuntimeServingExecution::Create(
                   std::move(*runtime), command)
                   .ok());

  runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok());
  ASSERT_TRUE(runtime->mark_ready().ok());
  ASSERT_TRUE(runtime->commit().ok());
  command.plan = runtime->descriptor();
  command.session.world_size = 2;
  EXPECT_FALSE(DeepSeekRankPlanRuntimeServingExecution::Create(
                   std::move(*runtime), command)
                   .ok());

  runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok());
  ASSERT_TRUE(runtime->mark_ready().ok());
  ASSERT_TRUE(runtime->commit().ok());
  command.session.world_size = 1;
  command.plan = runtime->descriptor();
  auto execution = DeepSeekRankPlanRuntimeServingExecution::Create(
      std::move(*runtime), command);
  ASSERT_TRUE(execution.ok()) << execution.status().message();
  ASSERT_TRUE(execution->cancel().ok());
  EXPECT_FALSE(execution->advance().value());
  EXPECT_TRUE(execution->advance().value());
}

TEST(DeepSeekRankPlanRuntimeTest, RejectsForgedReadyAndDuplicateCommit) {
  RuntimeFixture fixture;
  ScriptedCompute* compute = nullptr;
  auto runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok());
  EXPECT_EQ(runtime->commit().code(), StatusCode::kFailedPrecondition);
  ASSERT_TRUE(runtime->mark_ready().ok());
  EXPECT_EQ(runtime->mark_ready().code(), StatusCode::kFailedPrecondition);
  EXPECT_FALSE(runtime->validate_commit().ok());
  ASSERT_TRUE(runtime->prepare_commit().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kReady);
  EXPECT_TRUE(runtime->validate_commit().ok());
  ASSERT_TRUE(runtime->commit().ok());
  EXPECT_EQ(runtime->commit().code(), StatusCode::kFailedPrecondition);
}

TEST(DeepSeekRankPlanRuntimeTest,
     StagedCompletionRetainsResourcesUntilExplicitComplete) {
  RuntimeFixture fixture;
  ScriptedCompute* compute = nullptr;
  auto runtime = fixture.create(compute);
  ASSERT_TRUE(runtime.ok());
  ASSERT_TRUE(runtime->mark_ready().ok());
  ASSERT_TRUE(runtime->commit().ok());
  ASSERT_TRUE(runtime->advance_staged().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kExecuting);
  ASSERT_TRUE(runtime->advance_staged().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kCompletionReady);
  EXPECT_TRUE(runtime->validate_complete().ok());
  EXPECT_GT(fixture.provider.active, 0U);
  ASSERT_TRUE(runtime->complete().ok());
  EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kComplete);
  EXPECT_EQ(fixture.provider.active, 0U);
}

TEST(DeepSeekRankPlanRuntimeTest,
     PoisonedCommittedExecutionRetainsResourcesUntilTeardown) {
  RuntimeFixture fixture;
  ScriptedCompute* compute = nullptr;
  {
    auto runtime = fixture.create(compute);
    ASSERT_TRUE(runtime.ok());
    compute->poll_result = DeepSeekStageComputeStatus::kError;
    ASSERT_TRUE(runtime->mark_ready().ok());
    ASSERT_TRUE(runtime->commit().ok());
    ASSERT_TRUE(runtime->advance().ok());
    EXPECT_FALSE(runtime->advance().ok());
    EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kPoisoned);
    EXPECT_GT(fixture.provider.active, 0U);
  }
  EXPECT_EQ(fixture.provider.active, 0U);
}

TEST(DeepSeekRankPlanRuntimeTest,
     BorrowedComputeDriverExecutesWithoutTakingOwnership) {
  RuntimeFixture fixture;
  ScriptedCompute compute;
  {
    auto runtime = fixture.create_borrowed(compute);
    ASSERT_TRUE(runtime.ok()) << runtime.status().message();
    ASSERT_TRUE(runtime->mark_ready().ok());
    ASSERT_TRUE(runtime->commit().ok());
    ASSERT_TRUE(runtime->advance().ok());
    ASSERT_TRUE(runtime->advance().ok());
    EXPECT_EQ(runtime->state(), DeepSeekRankPlanRuntimeState::kComplete);
  }
  EXPECT_EQ(compute.launches, 1U);
  EXPECT_EQ(compute.polls, 1U);
}

TEST(DeepSeekRankPlanRuntimeTest,
     BoundaryDriverSetHasOneOwnerForAllBorrowedInterfaces) {
  DeepSeekOwnedBoundaryDriverSet boundary;
  boundary.owner = std::make_unique<CombinedBoundaryOwner>();
  ASSERT_NE(boundary.owner, nullptr);
  EXPECT_EQ(dynamic_cast<void*>(&boundary.owner->binding()),
            dynamic_cast<void*>(&boundary.owner->nccl()));
  EXPECT_EQ(dynamic_cast<void*>(&boundary.owner->binding()),
            dynamic_cast<void*>(&boundary.owner->event()));
  EXPECT_EQ(dynamic_cast<void*>(&boundary.owner->binding()),
            dynamic_cast<void*>(&boundary.owner->evidence()));
}

}  // namespace
}  // namespace pih
