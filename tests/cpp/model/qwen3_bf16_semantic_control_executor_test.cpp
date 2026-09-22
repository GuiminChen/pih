#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_semantic_control_executor.h"

namespace pih {
namespace {

QwenKvSlotPool ready_pool() {
  constexpr std::uint32_t slots = 257;
  auto pool = QwenKvSlotPool::Create(
      slots, slots * QwenKvSlotPool::kSlotPayloadBytes,
      slots * sizeof(QwenKvSlotState)).value();
  EXPECT_TRUE(pool.complete_startup_sanitize(
      slots * QwenKvSlotPool::kSlotPayloadBytes,
      slots * sizeof(QwenKvSlotState), true).ok());
  return pool;
}

std::vector<std::int64_t> manifest() {
  return std::vector<std::int64_t>(
      QwenBf16TapFixtureSequence::kRequiredTokenCount, 7);
}

class Backend final : public QwenBf16SequenceBackend,
                      public QwenBf16CompletionIdentityProvider {
 public:
  Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& table,
      const QwenKvAppendPlan& append) override {
    first_positions.push_back(first_position);
    committed_before.push_back(table.descriptor().committed_tokens);
    targets.push_back(append.target_committed_tokens);
    if (calls == fail_at) {
      ++calls;
      return Status::Internal("injected control failure");
    }
    ++calls;
    last = {91, static_cast<std::uint64_t>(100 + calls)};
    return static_cast<std::int64_t>(1000 + tokens.size());
  }
  Result<QwenKvCompletionEvent> last_completion_event() const override {
    if (invalid_completion) return QwenKvCompletionEvent{};
    return last;
  }
  std::size_t calls = 0;
  std::size_t fail_at = 99;
  bool invalid_completion = false;
  QwenKvCompletionEvent last{};
  std::vector<std::uint64_t> first_positions;
  std::vector<std::uint32_t> committed_before;
  std::vector<std::uint32_t> targets;
};

TEST(QwenBf16SemanticControlExecutorTest,
     RunsFixedSequenceWithoutTapAndSealsSemanticState) {
  auto pool = ready_pool();
  const auto suite = QwenBf16TapSuitePlan::Create().value();
  const auto sequence =
      QwenBf16TapFixtureSequence::Create(manifest(), suite).value();
  Backend backend;
  auto executor = QwenBf16SemanticControlExecutor::Admit(
      pool, 3, 9, sequence, backend, backend).value();
  ASSERT_TRUE(executor.run().ok());
  EXPECT_EQ(backend.calls, 5);
  EXPECT_EQ(backend.first_positions,
            (std::vector<std::uint64_t>{0, 1, 3, 18, 130}));
  EXPECT_EQ(backend.committed_before,
            (std::vector<std::uint32_t>{0, 1, 3, 18, 130}));
  EXPECT_EQ(backend.targets,
            (std::vector<std::uint32_t>{1, 3, 18, 130, 4098}));
  EXPECT_EQ(executor.committed_tokens(), 4098);
  QwenSemanticOutcomeRecorder recorder;
  EXPECT_TRUE(executor.publish_token_semantics(recorder).ok());
  EXPECT_FALSE(executor.publish_token_semantics(recorder).ok());
  auto plan = executor.make_kv_observation_plan(
      257 * QwenKvSlotPool::kSlotPayloadBytes);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->slices().size(), 257 * 2 * 28);
  EXPECT_TRUE(executor.release().ok());
}

TEST(QwenBf16SemanticControlExecutorTest,
     RollsBackUncommittedFixtureAndPoisonsOnBackendFailure) {
  auto pool = ready_pool();
  const auto suite = QwenBf16TapSuitePlan::Create().value();
  const auto sequence =
      QwenBf16TapFixtureSequence::Create(manifest(), suite).value();
  Backend backend;
  backend.fail_at = 2;
  auto executor = QwenBf16SemanticControlExecutor::Admit(
      pool, 3, 9, sequence, backend, backend).value();
  EXPECT_FALSE(executor.run().ok());
  EXPECT_EQ(executor.committed_tokens(), 3);
  EXPECT_EQ(executor.state(),
            QwenBf16SemanticControlExecutorState::kPoisoned);
  EXPECT_TRUE(executor.release().ok());
}

TEST(QwenBf16SemanticControlExecutorTest,
     InvalidCompletionCannotPublishKvCommit) {
  auto pool = ready_pool();
  const auto suite = QwenBf16TapSuitePlan::Create().value();
  const auto sequence =
      QwenBf16TapFixtureSequence::Create(manifest(), suite).value();
  Backend backend;
  backend.invalid_completion = true;
  auto executor = QwenBf16SemanticControlExecutor::Admit(
      pool, 3, 9, sequence, backend, backend).value();
  EXPECT_FALSE(executor.run().ok());
  EXPECT_EQ(executor.committed_tokens(), 0);
}

}  // namespace
}  // namespace pih
