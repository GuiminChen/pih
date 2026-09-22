#include "pih/model/deepseek_rank_serving_protocol.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest seal_root() {
  auto root = Sha256Digest::ParseHex(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
  return root.value();
}

DeepSeekRankServingSessionBinding binding(std::uint32_t rank,
                                          std::uint32_t world_size = 2) {
  return {41, 7, world_size, rank, 900 + rank, 950 + rank, 1000 + rank,
          seal_root()};
}

DeepSeekRankServingCommand execute(DeepSeekRankServingSessionBinding session,
                                   std::uint64_t sequence,
                                   std::uint64_t request_id = 61) {
  DeepSeekRankServingCommand command;
  command.session = session;
  command.command_sequence = sequence;
  command.request_id = request_id;
  command.request_generation = 3;
  command.plan = {41, 19, DeepSeekPlanPhase::kDecode, 1, 1};
  command.input_lease_identity = session.rank == 0 ? 71 : 0;
  command.output_lease_identity = session.rank + 1 == session.world_size ? 81 : 0;
  return command;
}

DeepSeekRankServingCompletion completion(
    const DeepSeekRankServingCommand& command,
    DeepSeekRankServingCompletionOutcome outcome =
        DeepSeekRankServingCompletionOutcome::kCompleted) {
  DeepSeekRankServingCompletion result;
  result.session = command.session;
  result.execution_command_sequence = command.command_sequence;
  result.request_id = command.request_id;
  result.request_generation = command.request_generation;
  result.plan = command.plan;
  result.outcome = outcome;
  result.output_lease_identity =
      outcome == DeepSeekRankServingCompletionOutcome::kCompleted
          ? command.output_lease_identity
          : 0;
  return result;
}

TEST(DeepSeekRankServingProtocolTest,
     BindsExecutionToWarmSealedSessionAndRankOwnedLeases) {
  auto session = binding(0);
  auto gate = DeepSeekRankServingWorkerGate::Create(session).value();
  auto command = execute(session, 1);

  EXPECT_EQ(kDeepSeekRankServingProtocolAbi,
            "pih_deepseek_rank_serving_protocol_v1");
  ASSERT_TRUE(gate.accept(command).ok());
  EXPECT_TRUE(gate.execution_active());
  EXPECT_FALSE(gate.cancellation_requested());
  ASSERT_TRUE(gate.complete(completion(command)).ok());
  EXPECT_FALSE(gate.execution_active());
  EXPECT_FALSE(gate.failed());
}

TEST(DeepSeekRankServingProtocolTest,
     EnforcesFirstAndLastRankLeaseOwnership) {
  auto first = binding(0);
  auto first_gate = DeepSeekRankServingWorkerGate::Create(first).value();
  auto malformed_first = execute(first, 1);
  malformed_first.input_lease_identity = 0;
  EXPECT_EQ(first_gate.accept(malformed_first).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(first_gate.failed());

  auto middle = binding(1, 3);
  auto middle_gate = DeepSeekRankServingWorkerGate::Create(middle).value();
  auto malformed_middle = execute(middle, 1);
  malformed_middle.output_lease_identity = 91;
  EXPECT_EQ(middle_gate.accept(malformed_middle).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(middle_gate.failed());
}

TEST(DeepSeekRankServingProtocolTest,
     CancellationSuppressesOutputAndBindsTheActiveGeneration) {
  auto session = binding(1);
  auto gate = DeepSeekRankServingWorkerGate::Create(session).value();
  auto command = execute(session, 1);
  ASSERT_TRUE(gate.accept(command).ok());

  auto cancel = command;
  cancel.command_sequence = 2;
  cancel.kind = DeepSeekRankServingCommandKind::kCancel;
  cancel.target_execution_sequence = command.command_sequence;
  cancel.input_lease_identity = 0;
  cancel.output_lease_identity = 0;
  ASSERT_TRUE(gate.accept(cancel).ok());
  EXPECT_TRUE(gate.cancellation_requested());

  auto published = completion(command);
  EXPECT_EQ(gate.complete(published).code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(gate.failed());

  auto second_gate = DeepSeekRankServingWorkerGate::Create(session).value();
  ASSERT_TRUE(second_gate.accept(command).ok());
  ASSERT_TRUE(second_gate.accept(cancel).ok());
  ASSERT_TRUE(second_gate.complete(
      completion(command, DeepSeekRankServingCompletionOutcome::kCancelled)).ok());
  EXPECT_FALSE(second_gate.execution_active());
  EXPECT_FALSE(second_gate.failed());
}

TEST(DeepSeekRankServingProtocolTest,
     RejectsStaleSessionAndNonMonotonicCommandSequences) {
  auto session = binding(0);
  auto gate = DeepSeekRankServingWorkerGate::Create(session).value();
  auto stale = execute(session, 1);
  stale.session.worker_generation++;
  EXPECT_EQ(gate.accept(stale).code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(gate.failed());

  auto second_gate = DeepSeekRankServingWorkerGate::Create(session).value();
  auto command = execute(session, 2);
  EXPECT_EQ(second_gate.accept(command).code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(second_gate.failed());
}

TEST(DeepSeekRankServingProtocolTest,
     RejectsDrainPlanBecauseCancellationOwnsServiceDrain) {
  auto session = binding(0);
  auto gate = DeepSeekRankServingWorkerGate::Create(session).value();
  auto command = execute(session, 1);
  command.plan.phase = DeepSeekPlanPhase::kDrain;
  command.plan.token_count = 0;

  EXPECT_EQ(gate.accept(command).code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(gate.failed());
}

}  // namespace
}  // namespace pih
