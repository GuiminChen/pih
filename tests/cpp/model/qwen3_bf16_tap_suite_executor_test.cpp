#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_tap_suite_executor.h"

namespace pih {
namespace {

QwenKvSlotPool ready_pool(std::uint32_t slots) {
  auto pool = QwenKvSlotPool::Create(
      slots, slots * QwenKvSlotPool::kSlotPayloadBytes,
      slots * sizeof(QwenKvSlotState)).value();
  EXPECT_TRUE(pool.complete_startup_sanitize(
      slots * QwenKvSlotPool::kSlotPayloadBytes,
      slots * sizeof(QwenKvSlotState), true).ok());
  return pool;
}

Sha256Digest digest(std::uint8_t value) {
  Sha256Digest result;
  result.bytes[0] = static_cast<std::byte>(value);
  return result;
}

class Backend final : public QwenBf16TapFixtureBackend {
 public:
  Result<QwenBf16TapFixtureExecutionReceipt> execute_and_capture(
      const QwenBf16TapFixtureInput& input,
      const QwenNumericalTapPlan& taps,
      const QwenKvBlockTable& table,
      const QwenKvAppendPlan& append,
      std::span<const QwenKvSlotState> projected_slots,
      std::uint64_t fixture_generation) override {
    first_positions.push_back(input.first_position);
    committed_before.push_back(table.descriptor().committed_tokens);
    targets.push_back(append.target_committed_tokens);
    const auto handle = table.reserved_handles()[
        input.capture_position / QwenKvSlotPool::kTokensPerSlot];
    projected_valid_tokens.push_back(projected_slots[handle.slot].valid_tokens);
    if (fail_at == calls++) return Status::Internal("injected tap failure");
    return QwenBf16TapFixtureExecutionReceipt{
        {fixture_generation, taps.captures().size(),
         taps.semantic_digest().value(), digest(static_cast<std::uint8_t>(calls))},
        invalid_completion ? QwenKvCompletionEvent{} :
                             QwenKvCompletionEvent{91, fixture_generation},
        static_cast<std::int64_t>(1000 + calls)};
  }

  std::size_t calls = 0;
  std::size_t fail_at = 99;
  bool invalid_completion = false;
  std::vector<std::uint32_t> first_positions;
  std::vector<std::uint32_t> committed_before;
  std::vector<std::uint32_t> targets;
  std::vector<std::uint16_t> projected_valid_tokens;
};

std::vector<std::int64_t> manifest() {
  return std::vector<std::int64_t>(
      QwenBf16TapFixtureSequence::kRequiredTokenCount, 7);
}

TEST(QwenBf16TapSuiteExecutorTest, CommitsFiveFixturesOnlyAfterCapture) {
  auto pool = ready_pool(257);
  auto suite = QwenBf16TapSuitePlan::Create().value();
  auto sequence = QwenBf16TapFixtureSequence::Create(manifest(), suite).value();
  Backend backend;
  auto executor = QwenBf16TapSuiteExecutor::Admit(
      pool, 3, 9, suite, sequence, 40, 100, backend);

  ASSERT_TRUE(executor.ok()) << executor.status().message();
  auto receipt = executor->run();
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->fixture_count, 5);
  EXPECT_EQ(receipt->capture_count, 369);
  EXPECT_EQ(executor->committed_tokens(), 4098);
  EXPECT_EQ(executor->last_completion_event().handle, 91);
  EXPECT_EQ(executor->last_completion_event().generation, 104);
  EXPECT_EQ(backend.first_positions,
            (std::vector<std::uint32_t>{0, 1, 3, 18, 130}));
  EXPECT_EQ(backend.committed_before,
            (std::vector<std::uint32_t>{0, 1, 3, 18, 130}));
  EXPECT_EQ(backend.targets,
            (std::vector<std::uint32_t>{1, 3, 18, 130, 4098}));
  EXPECT_EQ(backend.projected_valid_tokens,
            (std::vector<std::uint16_t>{1, 3, 2, 2, 2}));
  QwenSemanticOutcomeRecorder recorder;
  EXPECT_TRUE(executor->publish_token_semantics(recorder).ok());
  EXPECT_FALSE(executor->publish_token_semantics(recorder).ok());
  auto kv_plan = executor->make_kv_observation_plan(
      257 * QwenKvSlotPool::kSlotPayloadBytes);
  ASSERT_TRUE(kv_plan.ok()) << kv_plan.status().message();
  EXPECT_EQ(kv_plan->payload_bytes(),
            4098 * QwenKvAddressMapper::kBytesPerToken * 2 * 28);
  EXPECT_EQ(kv_plan->slices().size(), 257 * 2 * 28);
  EXPECT_TRUE(executor->release().ok());
  EXPECT_FALSE(executor->make_kv_observation_plan(
                   257 * QwenKvSlotPool::kSlotPayloadBytes)
                   .ok());
}

TEST(QwenBf16TapSuiteExecutorTest, FailureRollsBackAndPoisons) {
  auto pool = ready_pool(257);
  auto suite = QwenBf16TapSuitePlan::Create().value();
  auto sequence = QwenBf16TapFixtureSequence::Create(manifest(), suite).value();
  Backend backend;
  backend.fail_at = 2;
  auto executor = QwenBf16TapSuiteExecutor::Admit(
      pool, 3, 9, suite, sequence, 40, 100, backend).value();

  EXPECT_FALSE(executor.run().ok());
  EXPECT_EQ(executor.committed_tokens(), 3);
  EXPECT_EQ(executor.state(), QwenBf16TapSuiteExecutorState::kPoisoned);
  EXPECT_FALSE(executor.run().ok());
  EXPECT_EQ(backend.calls, 3);
  EXPECT_TRUE(executor.release().ok());
}

TEST(QwenBf16TapSuiteExecutorTest, EmptyCompletionCannotCommitKv) {
  auto pool = ready_pool(257);
  auto suite = QwenBf16TapSuitePlan::Create().value();
  auto sequence = QwenBf16TapFixtureSequence::Create(manifest(), suite).value();
  Backend backend;
  backend.invalid_completion = true;
  auto executor = QwenBf16TapSuiteExecutor::Admit(
      pool, 3, 9, suite, sequence, 40, 100, backend).value();

  EXPECT_FALSE(executor.run().ok());
  EXPECT_EQ(executor.committed_tokens(), 0);
  EXPECT_EQ(executor.state(), QwenBf16TapSuiteExecutorState::kPoisoned);
}

}  // namespace
}  // namespace pih
