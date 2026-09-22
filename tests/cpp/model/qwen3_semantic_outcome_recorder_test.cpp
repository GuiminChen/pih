#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_outcome_recorder.h"

namespace pih {
namespace {

std::array<std::byte, 3> bytes(std::byte value) {
  return {value, value, value};
}

Qwen3Config official_config() {
  return Qwen3Config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                     1'000'000.0, 0.000001, 151643, 151645};
}

Status record_all(QwenSemanticOutcomeRecorder& recorder, std::byte value) {
  const auto payload = bytes(value);
  Status status = recorder.record_dispatch(payload);
  if (status.ok()) status = recorder.record_final_logits(payload);
  if (status.ok()) status = recorder.record_greedy_trajectory(payload);
  if (status.ok()) status = recorder.record_kv_state(payload);
  if (status.ok()) status = recorder.record_accepted_token_ledger(payload);
  return status;
}

TEST(QwenSemanticOutcomeRecorderTest, SealsDomainSeparatedRootsAndPeakCapacity) {
  QwenSemanticOutcomeRecorder first;
  ASSERT_TRUE(record_all(first, std::byte{1}).ok());
  ASSERT_TRUE(first.observe_capacity(100, 20).ok());
  ASSERT_TRUE(first.observe_capacity(90, 30).ok());

  auto outcome = first.seal();

  ASSERT_TRUE(outcome.ok()) << outcome.status().message();
  EXPECT_EQ(outcome->device_peak_bytes, 100);
  EXPECT_EQ(outcome->pinned_peak_bytes, 30);
  EXPECT_NE(outcome->dispatch_root, outcome->final_logits_root);
  EXPECT_NE(outcome->final_logits_root, outcome->kv_state_root);
  EXPECT_EQ(first.state(), QwenSemanticOutcomeRecorderState::kSealed);

  QwenSemanticOutcomeRecorder second;
  ASSERT_TRUE(record_all(second, std::byte{1}).ok());
  ASSERT_TRUE(second.observe_capacity(100, 30).ok());
  auto control = second.seal();
  ASSERT_TRUE(control.ok());
  EXPECT_TRUE(QwenSemanticControlReceipt::Create(
                  QwenNumericalRunIdentity::Parse(
                      "{\"schema\":\"pih.qwen_bf16_numerical_run.v1\","
                      "\"model_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
                      "\"fixture_sha256\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
                      "\"tolerance_sha256\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\","
                      "\"kernel_bundle_sha256\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\","
                      "\"build_sha256\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\","
                      "\"environment_sha256\":\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\","
                      "\"target_gpu\":\"RTX_4090_D\",\"target_sm\":\"sm_89\",\"run_generation\":7}")
                      .value(),
                  {7, 5, 369, outcome->dispatch_root, outcome->kv_state_root},
                  *outcome, *control)
                  .ok());
}

TEST(QwenSemanticOutcomeRecorderTest, IncompleteDuplicateAndInvalidWritesPoison) {
  QwenSemanticOutcomeRecorder incomplete;
  EXPECT_FALSE(incomplete.seal().ok());
  EXPECT_EQ(incomplete.state(), QwenSemanticOutcomeRecorderState::kPoisoned);

  QwenSemanticOutcomeRecorder duplicate;
  const auto payload = bytes(std::byte{2});
  ASSERT_TRUE(duplicate.record_dispatch(payload).ok());
  EXPECT_FALSE(duplicate.record_dispatch(payload).ok());
  EXPECT_EQ(duplicate.state(), QwenSemanticOutcomeRecorderState::kPoisoned);

  QwenSemanticOutcomeRecorder empty;
  EXPECT_FALSE(empty.record_final_logits({}).ok());
  EXPECT_EQ(empty.state(), QwenSemanticOutcomeRecorderState::kPoisoned);
}

TEST(QwenSemanticOutcomeRecorderTest, SemanticByteDriftChangesOnlyItsRoot) {
  QwenSemanticOutcomeRecorder first;
  QwenSemanticOutcomeRecorder second;
  ASSERT_TRUE(record_all(first, std::byte{3}).ok());
  ASSERT_TRUE(record_all(second, std::byte{3}).ok());
  ASSERT_TRUE(first.observe_capacity(10, 10).ok());
  ASSERT_TRUE(second.observe_capacity(10, 10).ok());
  auto lhs = first.seal();
  auto rhs = second.seal();
  ASSERT_TRUE(lhs.ok() && rhs.ok());
  EXPECT_EQ(lhs->dispatch_root, rhs->dispatch_root);

  QwenSemanticOutcomeRecorder drifted;
  const auto common = bytes(std::byte{3});
  const auto drift = bytes(std::byte{4});
  ASSERT_TRUE(drifted.record_dispatch(common).ok());
  ASSERT_TRUE(drifted.record_final_logits(drift).ok());
  ASSERT_TRUE(drifted.record_greedy_trajectory(common).ok());
  ASSERT_TRUE(drifted.record_kv_state(common).ok());
  ASSERT_TRUE(drifted.record_accepted_token_ledger(common).ok());
  ASSERT_TRUE(drifted.observe_capacity(10, 10).ok());
  auto changed = drifted.seal();
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(lhs->dispatch_root, changed->dispatch_root);
  EXPECT_NE(lhs->final_logits_root, changed->final_logits_root);
}

TEST(QwenSemanticOutcomeRecorderTest, RecordsCanonicalProductionCommands) {
  auto schedule = QwenBf16ExecutionSchedule::Create(official_config());
  ASSERT_TRUE(schedule.ok());
  auto weights = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(weights.ok());
  auto commands = QwenBf16CommandBuffer::Create(*schedule, *weights);
  ASSERT_TRUE(commands.ok());

  QwenSemanticOutcomeRecorder typed;
  QwenSemanticOutcomeRecorder raw;
  ASSERT_TRUE(typed.record_dispatch(*commands).ok());
  const auto wire = commands->canonical_wire();
  ASSERT_TRUE(raw.record_dispatch(wire).ok());

  const auto payload = bytes(std::byte{5});
  for (auto* recorder : {&typed, &raw}) {
    ASSERT_TRUE(recorder->record_final_logits(payload).ok());
    ASSERT_TRUE(recorder->record_greedy_trajectory(payload).ok());
    ASSERT_TRUE(recorder->record_kv_state(payload).ok());
    ASSERT_TRUE(recorder->record_accepted_token_ledger(payload).ok());
    ASSERT_TRUE(recorder->observe_capacity(100, 20).ok());
  }
  auto typed_outcome = typed.seal();
  auto raw_outcome = raw.seal();
  ASSERT_TRUE(typed_outcome.ok() && raw_outcome.ok());
  EXPECT_EQ(typed_outcome->dispatch_root, raw_outcome->dispatch_root);
}

}  // namespace
}  // namespace pih
