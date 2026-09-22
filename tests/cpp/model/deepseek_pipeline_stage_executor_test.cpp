#include "pih/model/deepseek_pipeline_stage_executor.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih {
namespace {

class NullBinding final : public DeepSeekNcclP2pBindingTarget {
 public:
  Status bind_p2p(DeepSeekNcclRole, void*, std::uint64_t,
                  std::uintptr_t) override {
    return Status::Ok();
  }
};

class NullNccl final : public DeepSeekNcclP2pDriver {
 public:
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

class NullEvent final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
};

class NullEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class FixedBoundaryClock final : public DeepSeekBoundaryRuntimeClock {
 public:
  Result<std::uint64_t> now_ns() override { return now; }
  std::uint64_t now = 200;
};

class NeverCompleteBoundary final : public DeepSeekBoundaryOperation {
 public:
  Status submit(DeepSeekNcclP2pBindingTarget&, DeepSeekNcclP2pDriver&,
                CompletionEventDriver&) override {
    state_ = DeepSeekBoundaryExecutorState::kDeviceInFlight;
    return Status::Ok();
  }
  Status poll_issue(DeepSeekNcclP2pDriver&, CompletionEventDriver&) override {
    return Status::Unavailable("still issuing");
  }
  Status poll_completion(DeepSeekNcclP2pDriver&, CompletionEventDriver&,
                         CompletionEvidenceProvider&) override {
    return Status::Unavailable("still running");
  }
  Status expire(std::uint64_t now_ns) override {
    if (now_ns < 200) return Status::Unavailable("before deadline");
    state_ = DeepSeekBoundaryExecutorState::kPoisoned;
    return Status::Internal("deadline expired");
  }
  DeepSeekBoundaryExecutorState state() const noexcept override {
    return state_;
  }

 private:
  DeepSeekBoundaryExecutorState state_ =
      DeepSeekBoundaryExecutorState::kReady;
};

class ScriptedBoundary final : public DeepSeekBoundaryOperation {
 public:
  explicit ScriptedBoundary(std::string name, std::vector<std::string>& trace)
      : name_(std::move(name)), trace_(trace) {}

  Status submit(DeepSeekNcclP2pBindingTarget&, DeepSeekNcclP2pDriver&,
                CompletionEventDriver&) override {
    trace_.push_back(name_ + ":submit");
    if (fail_) {
      state_ = DeepSeekBoundaryExecutorState::kPoisoned;
      return Status::Internal(name_ + " failed");
    }
    state_ = DeepSeekBoundaryExecutorState::kDeviceInFlight;
    return Status::Ok();
  }
  Status poll_issue(DeepSeekNcclP2pDriver&,
                    CompletionEventDriver&) override {
    return Status::Internal("unexpected issue poll");
  }
  Status poll_completion(DeepSeekNcclP2pDriver&, CompletionEventDriver&,
                         CompletionEvidenceProvider&) override {
    trace_.push_back(name_ + ":complete");
    state_ = DeepSeekBoundaryExecutorState::kCompleteVerified;
    return Status::Ok();
  }
  DeepSeekBoundaryExecutorState state() const noexcept override {
    return state_;
  }
  void fail_on_submit() noexcept { fail_ = true; }

 private:
  std::string name_;
  std::vector<std::string>& trace_;
  DeepSeekBoundaryExecutorState state_ =
      DeepSeekBoundaryExecutorState::kReady;
  bool fail_ = false;
};

class ScriptedCompute final : public DeepSeekStageComputeDriver {
 public:
  explicit ScriptedCompute(std::vector<std::string>& trace) : trace_(trace) {}
  Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                const DeepSeekStagePlan& stage) override {
    trace_.push_back("compute:launch");
    launched_rank = stage.rank;
    launched_sequence = plan.plan_sequence;
    return launch_status;
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    trace_.push_back("compute:poll");
    return poll_status;
  }
  std::uint32_t launched_rank = 99;
  std::uint64_t launched_sequence = 0;
  Status launch_status = Status::Ok();
  DeepSeekStageComputeStatus poll_status = DeepSeekStageComputeStatus::kSuccess;

 private:
  std::vector<std::string>& trace_;
};

DeepSeekStageExecutionDrivers drivers_for(ScriptedCompute& compute,
                                           NullBinding& binding,
                                           NullNccl& nccl, NullEvent& event,
                                           NullEvidence& evidence) {
  DeepSeekBoundaryDriverSet boundary{&binding, &nccl, &event, &evidence};
  return {&compute, boundary, boundary};
}

Result<DeepSeekPipelineTransaction> prepared_transaction(
    DeepSeekPipelineResourceSet& resources, std::uint32_t world_size) {
  auto capacity = DeepSeekPipelineCapacity::Create(world_size, 8, 8, 1, false);
  if (!capacity.ok()) return capacity.status();
  auto created = DeepSeekPipelineResourceSet::Create(*capacity);
  if (!created.ok()) return created.status();
  resources = std::move(*created);
  auto transaction = resources.prepare(
      {9, 1, DeepSeekPlanPhase::kDecode, 2, 2});
  return transaction;
}

Result<DeepSeekPipelineTransaction> committed_transaction(
    DeepSeekPipelineResourceSet& resources, std::uint32_t world_size) {
  auto transaction = prepared_transaction(resources, world_size);
  if (!transaction.ok()) return transaction.status();
  const auto status = transaction->commit();
  if (!status.ok()) return status;
  return std::move(*transaction);
}

TEST(DeepSeekPipelineStageExecutorTest,
     StagesPreparedTransactionWithoutWeakeningCreateContract) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = prepared_transaction(resources, 1);
  auto plan = DeepSeekPipelinePlan::Create(1, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  EXPECT_FALSE(DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(0), 1, nullptr, nullptr).ok());
  auto staged = DeepSeekPipelineStageExecutor::CreateStaged(
      *transaction, plan->rank(0), 1, nullptr, nullptr);
  ASSERT_TRUE(staged.ok()) << staged.status().message();
  EXPECT_EQ(transaction->state(), DeepSeekPipelineTransactionState::kPrepared);
  EXPECT_EQ(staged->state(), DeepSeekPipelineStageExecutorState::kReady);
}

TEST(DeepSeekPipelineStageExecutorTest, RunsSingleRankComputeToCompletion) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = committed_transaction(resources, 1);
  auto plan = DeepSeekPipelinePlan::Create(1, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  auto executor = DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(0), 1, nullptr, nullptr);
  ASSERT_TRUE(executor.ok()) << executor.status().message();
  std::vector<std::string> trace;
  ScriptedCompute compute(trace);
  NullBinding binding; NullNccl nccl; NullEvent event; NullEvidence evidence;
  auto drivers = drivers_for(compute, binding, nccl, event, evidence);

  ASSERT_TRUE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->state(), DeepSeekPipelineStageExecutorState::kCompute);
  ASSERT_TRUE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->state(), DeepSeekPipelineStageExecutorState::kComplete);
  EXPECT_EQ(trace, (std::vector<std::string>{"compute:launch", "compute:poll"}));
  EXPECT_EQ(compute.launched_rank, 0U);
  EXPECT_EQ(compute.launched_sequence, 1U);
}

TEST(DeepSeekPipelineStageExecutorTest, OrdersMiddleRankIncomingComputeOutgoing) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = committed_transaction(resources, 3);
  auto plan = DeepSeekPipelinePlan::Create(3, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  std::vector<std::string> trace;
  ScriptedBoundary incoming("incoming", trace);
  ScriptedBoundary outgoing("outgoing", trace);
  auto executor = DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(1), 3, &incoming, &outgoing);
  ASSERT_TRUE(executor.ok()) << executor.status().message();
  ScriptedCompute compute(trace);
  NullBinding binding; NullNccl nccl; NullEvent event; NullEvidence evidence;
  auto drivers = drivers_for(compute, binding, nccl, event, evidence);

  EXPECT_EQ(executor->advance(drivers).code(), StatusCode::kUnavailable);
  EXPECT_TRUE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->advance(drivers).code(), StatusCode::kUnavailable);
  EXPECT_TRUE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->state(), DeepSeekPipelineStageExecutorState::kComplete);
  EXPECT_EQ(trace, (std::vector<std::string>{
      "incoming:submit", "incoming:complete", "compute:launch",
      "compute:poll", "outgoing:submit", "outgoing:complete"}));
}

TEST(DeepSeekPipelineStageExecutorTest, RejectsBoundaryTopologyMismatch) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = committed_transaction(resources, 2);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  std::vector<std::string> trace;
  ScriptedBoundary boundary("boundary", trace);
  EXPECT_FALSE(DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(0), 2, &boundary, &boundary).ok());
  EXPECT_FALSE(DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(1), 2, nullptr, nullptr).ok());
}

TEST(DeepSeekPipelineStageExecutorTest, BoundaryFailurePoisonsStage) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = committed_transaction(resources, 2);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  std::vector<std::string> trace;
  ScriptedBoundary incoming("incoming", trace);
  incoming.fail_on_submit();
  auto executor = DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(1), 2, &incoming, nullptr);
  ASSERT_TRUE(executor.ok());
  ScriptedCompute compute(trace);
  NullBinding binding; NullNccl nccl; NullEvent event; NullEvidence evidence;
  auto drivers = drivers_for(compute, binding, nccl, event, evidence);
  EXPECT_FALSE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->state(), DeepSeekPipelineStageExecutorState::kPoisoned);
}

TEST(DeepSeekPipelineStageExecutorTest, BoundaryDeadlinePoisonsStage) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = committed_transaction(resources, 2);
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  NeverCompleteBoundary incoming;
  auto executor = DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(1), 2, &incoming, nullptr);
  ASSERT_TRUE(executor.ok());
  std::vector<std::string> trace;
  ScriptedCompute compute(trace);
  NullBinding binding; NullNccl nccl; NullEvent event; NullEvidence evidence;
  FixedBoundaryClock clock;
  auto drivers = drivers_for(compute, binding, nccl, event, evidence);
  drivers.incoming.clock = &clock;

  EXPECT_EQ(executor->advance(drivers).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->state(), DeepSeekPipelineStageExecutorState::kPoisoned);
  EXPECT_EQ(incoming.state(), DeepSeekBoundaryExecutorState::kPoisoned);
}

TEST(DeepSeekPipelineStageExecutorTest, ComputeFailurePoisonsStage) {
  DeepSeekPipelineResourceSet resources;
  auto transaction = committed_transaction(resources, 1);
  auto plan = DeepSeekPipelinePlan::Create(1, false);
  ASSERT_TRUE(transaction.ok()); ASSERT_TRUE(plan.ok());
  auto executor = DeepSeekPipelineStageExecutor::Create(
      *transaction, plan->rank(0), 1, nullptr, nullptr);
  ASSERT_TRUE(executor.ok());
  std::vector<std::string> trace;
  ScriptedCompute compute(trace);
  compute.poll_status = DeepSeekStageComputeStatus::kError;
  NullBinding binding; NullNccl nccl; NullEvent event; NullEvidence evidence;
  auto drivers = drivers_for(compute, binding, nccl, event, evidence);
  ASSERT_TRUE(executor->advance(drivers).ok());
  EXPECT_FALSE(executor->advance(drivers).ok());
  EXPECT_EQ(executor->state(), DeepSeekPipelineStageExecutorState::kPoisoned);
  EXPECT_EQ(executor->advance(drivers).code(), StatusCode::kInternal);
}

}  // namespace
}  // namespace pih
