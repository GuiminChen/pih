#include "pih/model/deepseek_rank_serving_plan_executor.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest plan_executor_root() {
  return Sha256Digest::ParseHex(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
      .value();
}

DeepSeekRankServingCommand execution_command() {
  DeepSeekRankServingCommand command;
  command.session = {7, 8, 1, 0, 1, 2, 3, plan_executor_root()};
  command.command_sequence = 1;
  command.request_id = 4;
  command.request_generation = 5;
  command.plan = {7, 6, DeepSeekPlanPhase::kDecode, 1, 1};
  command.input_lease_identity = 8;
  command.output_lease_identity = 9;
  return command;
}

DeepSeekRankServingCommand cancellation_command(
    const DeepSeekRankServingCommand& execution) {
  auto command = execution;
  command.command_sequence = 2;
  command.kind = DeepSeekRankServingCommandKind::kCancel;
  command.target_execution_sequence = execution.command_sequence;
  command.input_lease_identity = 0;
  command.output_lease_identity = 0;
  return command;
}

class FakeExecution final : public DeepSeekRankServingPlanExecution {
 public:
  Status cancel() override {
    ++cancellations;
    return cancel_status;
  }
  Result<bool> advance() override {
    ++advances;
    return terminal;
  }

  Status cancel_status = Status::Ok();
  bool terminal = false;
  int cancellations = 0;
  int advances = 0;
};

class FakeFactory final : public DeepSeekRankServingPlanExecutionFactory {
 public:
  Result<std::unique_ptr<DeepSeekRankServingPlanExecution>> start(
      const DeepSeekRankServingCommand& command,
      DeepSeekRankServingResolvedLeases) override {
    ++starts;
    received = command;
    auto execution = std::make_unique<FakeExecution>();
    last = execution.get();
    return std::unique_ptr<DeepSeekRankServingPlanExecution>(
        std::move(execution));
  }

  int starts = 0;
  DeepSeekRankServingCommand received;
  FakeExecution* last = nullptr;
};

class FakeResolver final : public DeepSeekRankServingLeaseResolver {
 public:
  Result<DeepSeekRankServingResolvedLeases> resolve(
      const DeepSeekRankServingCommand& command) override {
    DeepSeekRankServingResolvedLeases leases;
    leases.input_lease_identity = command.input_lease_identity;
    leases.output_lease_identity = command.output_lease_identity;
    if (command.input_lease_identity != 0) leases.input_owner = std::make_shared<int>(1);
    if (command.output_lease_identity != 0) leases.output_owner = std::make_shared<int>(2);
    return leases;
  }
};

TEST(DeepSeekRankServingPlanWorkerExecutorTest,
     CompletesWithTheExactOutputLease) {
  FakeFactory factory;
  FakeResolver resolver;
  auto executor =
      DeepSeekRankServingPlanWorkerExecutor::Create(factory, resolver).value();
  const auto command = execution_command();
  ASSERT_TRUE(executor.start(command).ok());
  ASSERT_EQ(factory.starts, 1);
  ASSERT_EQ(factory.received.command_sequence, command.command_sequence);

  EXPECT_FALSE(executor.poll().value().has_value());
  factory.last->terminal = true;
  auto completion = executor.poll().value();
  ASSERT_TRUE(completion.has_value());
  EXPECT_EQ(completion->outcome,
            DeepSeekRankServingCompletionOutcome::kCompleted);
  EXPECT_EQ(completion->output_lease_identity, command.output_lease_identity);
  EXPECT_FALSE(executor.execution_active());
}

TEST(DeepSeekRankServingPlanWorkerExecutorTest,
     CancellationDrainsWithoutPublishingOutput) {
  FakeFactory factory;
  FakeResolver resolver;
  auto executor =
      DeepSeekRankServingPlanWorkerExecutor::Create(factory, resolver).value();
  const auto command = execution_command();
  ASSERT_TRUE(executor.start(command).ok());
  ASSERT_TRUE(executor.cancel(cancellation_command(command)).ok());
  EXPECT_EQ(factory.last->cancellations, 1);
  factory.last->terminal = true;

  auto completion = executor.poll().value();
  ASSERT_TRUE(completion.has_value());
  EXPECT_EQ(completion->outcome,
            DeepSeekRankServingCompletionOutcome::kCancelled);
  EXPECT_EQ(completion->output_lease_identity, 0);
}

TEST(DeepSeekRankServingPlanWorkerExecutorTest,
     RejectsForeignCancellationBeforeTouchingExecution) {
  FakeFactory factory;
  FakeResolver resolver;
  auto executor =
      DeepSeekRankServingPlanWorkerExecutor::Create(factory, resolver).value();
  const auto command = execution_command();
  ASSERT_TRUE(executor.start(command).ok());
  auto cancel = cancellation_command(command);
  cancel.target_execution_sequence = 99;

  EXPECT_EQ(executor.cancel(cancel).code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(factory.last->cancellations, 0);
}

TEST(DeepSeekRankServingPlanWorkerExecutorTest,
     RejectsMismatchedLeaseBeforeStartingFactory) {
  class MismatchedResolver final : public DeepSeekRankServingLeaseResolver {
   public:
    Result<DeepSeekRankServingResolvedLeases> resolve(
        const DeepSeekRankServingCommand& command) override {
      DeepSeekRankServingResolvedLeases leases;
      leases.input_lease_identity = command.input_lease_identity + 1;
      leases.output_lease_identity = command.output_lease_identity;
      leases.input_owner = std::make_shared<int>(1);
      leases.output_owner = std::make_shared<int>(2);
      return leases;
    }
  } resolver;
  FakeFactory factory;
  auto executor =
      DeepSeekRankServingPlanWorkerExecutor::Create(factory, resolver).value();
  EXPECT_EQ(executor.start(execution_command()).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(factory.starts, 0);
}

}  // namespace
}  // namespace pih
