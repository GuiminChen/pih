#include "pih/model/deepseek_pipeline_boundary_executor.h"
#include "pih/model/deepseek_prepared_boundary_operation.h"
#include "pih/model/deepseek_tracked_boundary_operation.h"

#include <gtest/gtest.h>

#include <new>

namespace pih {
namespace {

class ExecutorAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    auto device = Device::Create(DeviceType::kCuda, 0);
    if (!device.ok()) return device.status();
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t{static_cast<std::size_t>(alignment)});
    return Allocation{data, bytes, alignment, 17, *device};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data,
                      std::align_val_t{static_cast<std::size_t>(allocation.alignment)});
  }
};

class BoundaryDriver final : public DeepSeekNcclP2pBindingTarget,
                             public DeepSeekNcclP2pDriver {
 public:
  Status bind_p2p(DeepSeekNcclRole, void* buffer, std::uint64_t bytes,
                  std::uintptr_t stream) override {
    if (fail_binding) return Status::Internal("binding failed");
    bound = buffer != nullptr && bytes == 65536 && stream == 88;
    return bound ? Status::Ok() : Status::InvalidArgument("wrong binding");
  }
  Status group_start() override { return bound ? Status::Ok() : Status::Internal("unbound"); }
  Status send(std::uint64_t count, std::uint32_t peer) override {
    return count == 32768 && peer == 1 ? Status::Ok()
                                       : Status::InvalidArgument("wrong send");
  }
  Status recv(std::uint64_t, std::uint32_t) override {
    return Status::Internal("unexpected recv");
  }
  Result<DeepSeekNcclAsyncStatus> group_end() override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  Result<DeepSeekNcclAsyncStatus> async_status() override {
    return DeepSeekNcclAsyncStatus::kSuccess;
  }
  bool fail_binding = false;
  bool bound = false;
};

class BoundaryEventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle stream) override {
    return stream == 88 ? Status::Ok() : Status::InvalidArgument("wrong stream");
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
};

class BoundaryEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

DeepSeekNcclP2pManifest executor_manifest() {
  return {.operation_plan_id = 10, .engine_epoch = 2,
          .communicator_generation = 3, .pipeline_plan_sequence = 1,
          .operation_ordinal = 5, .global_issue_ordinal = 5,
          .directed_boundary_id = 0,
          .role = DeepSeekNcclRole::kSend, .local_global_rank = 0,
          .peer_global_rank = 1, .communicator_local_rank = 0,
          .communicator_peer_rank = 1, .buffer_owner_id = 9,
          .buffer_offset_bytes = 256, .buffer_capacity_bytes = 131072,
          .buffer_generation = 17, .context_identity = 77, .token_count = 2};
}

TEST(DeepSeekPipelineBoundaryExecutorTest, JoinsCommittedPlanThroughCompletion) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  ASSERT_TRUE(capacity.ok());
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(resources.ok());
  auto transaction = resources->prepare({2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->commit().ok());
  auto operation = DeepSeekNcclP2pPlan::Create(executor_manifest());
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  auto event = CompletionEventSlot::Create(1, 77);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 0, 1, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ASSERT_TRUE(operation.ok()); ASSERT_TRUE(sequencer.ok());
  ASSERT_TRUE(event.ok()); ASSERT_TRUE(frontier.ok());
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  ASSERT_TRUE(buffer.ok());
  auto lease = DeepSeekNcclBoundaryLease::Create(
      executor_manifest(), *buffer, 9, 77, 88);
  ASSERT_TRUE(lease.ok());
  auto executor = DeepSeekPipelineBoundaryExecutor::Create(
      *transaction, *operation, *lease, *sequencer, *event, *frontier);
  ASSERT_TRUE(executor.ok()) << executor.status().message();
  BoundaryDriver nccl;
  BoundaryEventDriver event_driver;
  BoundaryEvidence evidence;
  ASSERT_TRUE(executor->submit(nccl, nccl, event_driver).ok());
  EXPECT_EQ(executor->state(), DeepSeekBoundaryExecutorState::kDeviceInFlight);
  ASSERT_TRUE(executor->poll_completion(nccl, event_driver, evidence).ok());
  EXPECT_EQ(executor->state(), DeepSeekBoundaryExecutorState::kCompleteVerified);
  EXPECT_EQ(sequencer->next_ordinal(), 6U);
}

TEST(DeepSeekPipelineBoundaryExecutorTest, RejectsBeforeDistributedCommit) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  ASSERT_TRUE(capacity.ok());
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  ASSERT_TRUE(resources.ok());
  auto transaction = resources->prepare({2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  ASSERT_TRUE(transaction.ok());
  auto operation = DeepSeekNcclP2pPlan::Create(executor_manifest());
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  auto event = CompletionEventSlot::Create(1, 77);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 0, 1, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  auto lease = DeepSeekNcclBoundaryLease::Create(
      executor_manifest(), *buffer, 9, 77, 88);
  ASSERT_TRUE(operation.ok()); ASSERT_TRUE(sequencer.ok());
  ASSERT_TRUE(event.ok()); ASSERT_TRUE(frontier.ok()); ASSERT_TRUE(lease.ok());
  EXPECT_FALSE(DeepSeekPipelineBoundaryExecutor::Create(
      *transaction, *operation, *lease, *sequencer, *event, *frontier).ok());
}

TEST(DeepSeekPipelineBoundaryExecutorTest, BindingFailurePoisonsRankSequencer) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  auto transaction = resources->prepare({2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(transaction->commit().ok());
  auto operation = DeepSeekNcclP2pPlan::Create(executor_manifest());
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  auto event = CompletionEventSlot::Create(1, 77);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 0, 1, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  auto lease = DeepSeekNcclBoundaryLease::Create(
      executor_manifest(), *buffer, 9, 77, 88);
  auto executor = DeepSeekPipelineBoundaryExecutor::Create(
      *transaction, *operation, *lease, *sequencer, *event, *frontier);
  ASSERT_TRUE(executor.ok());
  BoundaryDriver nccl; nccl.fail_binding = true;
  BoundaryEventDriver event_driver;
  EXPECT_FALSE(executor->submit(nccl, nccl, event_driver).ok());
  EXPECT_EQ(executor->state(), DeepSeekBoundaryExecutorState::kPoisoned);
  EXPECT_TRUE(sequencer->poisoned());
}

TEST(DeepSeekPipelineBoundaryExecutorTest,
     DeadlineExpiryPoisonsBoundaryAndRankSequencer) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  auto transaction = resources->prepare({2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  ASSERT_TRUE(transaction.ok());
  ASSERT_TRUE(transaction->commit().ok());
  auto operation = DeepSeekNcclP2pPlan::Create(executor_manifest());
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  auto event = CompletionEventSlot::Create(1, 77);
  auto frontier = CudaCompletionFrontier::Create(
      {2, 0, 1, CudaCompletionPhase::kCopy, 5}, 5, 100, 200);
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  auto lease = DeepSeekNcclBoundaryLease::Create(
      executor_manifest(), *buffer, 9, 77, 88);
  auto executor = DeepSeekPipelineBoundaryExecutor::Create(
      *transaction, *operation, *lease, *sequencer, *event, *frontier);
  ASSERT_TRUE(executor.ok());

  EXPECT_EQ(executor->expire(199).code(), StatusCode::kUnavailable);
  const auto expired = executor->expire(200);
  EXPECT_FALSE(expired.ok());
  EXPECT_EQ(executor->state(), DeepSeekBoundaryExecutorState::kPoisoned);
  EXPECT_TRUE(sequencer->poisoned());
  EXPECT_EQ(frontier->first_failure(), CudaFrontierFailure::kDeadlineExpired);
}

TEST(DeepSeekPipelineBoundaryExecutorTest,
     PreparedOperationDefersExecutorUntilDistributedCommit) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  auto transaction = resources->prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(sequencer.ok());
  ASSERT_TRUE(buffer.ok());
  auto operation = DeepSeekPreparedBoundaryOperation::Create(
      *transaction, executor_manifest(), *buffer, 9, 77, 88, 1,
      *sequencer, 100, 200);
  ASSERT_TRUE(operation.ok()) << operation.status().message();
  EXPECT_EQ(operation->state(), DeepSeekBoundaryExecutorState::kReady);
  BoundaryDriver nccl;
  BoundaryEventDriver event_driver;
  BoundaryEvidence evidence;
  EXPECT_FALSE(operation->submit(nccl, nccl, event_driver).ok());
  EXPECT_EQ(operation->state(), DeepSeekBoundaryExecutorState::kReady);

  ASSERT_TRUE(transaction->commit().ok());
  ASSERT_TRUE(operation->submit(nccl, nccl, event_driver).ok());
  ASSERT_TRUE(operation->poll_completion(nccl, event_driver, evidence).ok());
  EXPECT_EQ(operation->state(),
            DeepSeekBoundaryExecutorState::kCompleteVerified);
  EXPECT_EQ(sequencer->next_ordinal(), 6U);
}

TEST(DeepSeekPipelineBoundaryExecutorTest,
     TrackedOperationReleasesOnlyAfterVerifiedCompletion) {
  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  auto transaction = resources->prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  auto credit = tracker.reserve(1, 5).value();
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  auto prepared = DeepSeekPreparedBoundaryOperation::Create(
      *transaction, executor_manifest(), *buffer, 9, 77, 88, 1,
      *sequencer, 100, 200);
  ASSERT_TRUE(prepared.ok());
  {
    auto tracked = DeepSeekTrackedBoundaryOperation::Create(
        std::make_unique<DeepSeekPreparedBoundaryOperation>(
            std::move(*prepared)), tracker, credit);
    ASSERT_TRUE(tracked.ok()) << tracked.status().message();
    ASSERT_TRUE(transaction->commit().ok());
    BoundaryDriver nccl;
    BoundaryEventDriver event_driver;
    BoundaryEvidence evidence;
    ASSERT_TRUE((*tracked)->submit(nccl, nccl, event_driver).ok());
    EXPECT_EQ(tracker.state(credit.credit_index),
              DeepSeekBoundaryCreditState::kCommitted);
    ASSERT_TRUE((*tracked)->poll_completion(
        nccl, event_driver, evidence).ok());
    EXPECT_EQ(tracker.state(credit.credit_index),
              DeepSeekBoundaryCreditState::kCompleteVerified);
  }
  EXPECT_EQ(tracker.state(credit.credit_index),
            DeepSeekBoundaryCreditState::kFree);
}

TEST(DeepSeekPipelineBoundaryExecutorTest,
     AllowsFutureOrdinalPreparationButSequencerStillRejectsEarlyIssue) {
  auto capacity = DeepSeekPipelineCapacity::Create(3, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  auto transaction = resources->prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  auto sequencer = DeepSeekNcclOperationSequencer::Create(1);
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  auto future = executor_manifest();
  future.operation_plan_id = 11;
  future.operation_ordinal = 2;
  future.global_issue_ordinal = 2;
  auto prepared = DeepSeekPreparedBoundaryOperation::Create(
      *transaction, future, *buffer, 9, 77, 88, 1,
      *sequencer, 100, 200);
  ASSERT_TRUE(prepared.ok()) << prepared.status().message();
  ASSERT_TRUE(transaction->commit().ok());
  BoundaryDriver nccl;
  BoundaryEventDriver event_driver;
  EXPECT_FALSE(prepared->submit(nccl, nccl, event_driver).ok());
  EXPECT_EQ(sequencer->next_ordinal(), 1U);
}

TEST(DeepSeekPipelineBoundaryExecutorTest,
     TrackedOperationAbortsPrepareButQuarantinesCommittedDestruction) {
  auto tracker = DeepSeekBoundaryCreditTracker::Create(2).value();
  {
    auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
    auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
    auto transaction = resources->prepare(
        {2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
    auto sequencer = DeepSeekNcclOperationSequencer::Create(5);
    auto credit = tracker.reserve(1, 5).value();
    ExecutorAllocator allocator;
    auto buffer = Buffer::Allocate(allocator, 131072, 256);
    auto prepared = DeepSeekPreparedBoundaryOperation::Create(
        *transaction, executor_manifest(), *buffer, 9, 77, 88, 1,
        *sequencer, 100, 200);
    auto tracked = DeepSeekTrackedBoundaryOperation::Create(
        std::make_unique<DeepSeekPreparedBoundaryOperation>(
            std::move(*prepared)), tracker, credit);
    ASSERT_TRUE(tracked.ok());
  }
  EXPECT_EQ(tracker.state(0), DeepSeekBoundaryCreditState::kFree);

  auto capacity = DeepSeekPipelineCapacity::Create(2, 8, 8, 1, false);
  auto resources = DeepSeekPipelineResourceSet::Create(*capacity);
  auto transaction = resources->prepare(
      {2, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  auto sequencer = DeepSeekNcclOperationSequencer::Create(6);
  auto manifest = executor_manifest();
  manifest.operation_plan_id = 11;
  manifest.operation_ordinal = 6;
  auto credit = tracker.reserve(1, 6).value();
  ExecutorAllocator allocator;
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  auto prepared = DeepSeekPreparedBoundaryOperation::Create(
      *transaction, manifest, *buffer, 9, 77, 88, 1, *sequencer, 100, 200);
  auto tracked = DeepSeekTrackedBoundaryOperation::Create(
      std::make_unique<DeepSeekPreparedBoundaryOperation>(
          std::move(*prepared)), tracker, credit);
  ASSERT_TRUE(tracked.ok());
  ASSERT_TRUE(transaction->commit().ok());
  BoundaryDriver nccl;
  BoundaryEventDriver event_driver;
  ASSERT_TRUE((*tracked)->submit(nccl, nccl, event_driver).ok());
  tracked->reset();
  EXPECT_TRUE(tracker.poisoned());
  EXPECT_EQ(tracker.state(credit.credit_index),
            DeepSeekBoundaryCreditState::kSuspect);
}

}  // namespace
}  // namespace pih
