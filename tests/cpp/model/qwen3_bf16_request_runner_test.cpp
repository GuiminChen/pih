#include "pih/model/qwen3_bf16_request_runner.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Backend final : public QwenBf16SequenceBackend {
 public:
  Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens, std::uint64_t,
      const QwenKvBlockTable&, const QwenKvAppendPlan& append_plan) override {
    calls.push_back(tokens.size());
    sequence_generations.push_back(
        append_plan.expected_sequence_generation);
    if (!status.ok()) return status;
    return calls.size() == 1 ? 7 : 2;
  }
  Status status = Status::Ok();
  std::vector<std::size_t> calls;
  std::vector<std::uint32_t> sequence_generations;
};

class Completion final : public QwenBf16CompletionIdentityProvider {
 public:
  Result<QwenKvCompletionEvent> last_completion_event() const override {
    return event;
  }
  QwenKvCompletionEvent event{31, 37};
};

class Recycler final : public QwenBf16KvRecycler {
 public:
  Status recycle(QwenKvSlotPool& pool,
                 std::span<const QwenKvBlockHandle> handles,
                 QwenKvCompletionEvent event) override {
    ++calls;
    seen_event = event;
    for (const auto handle : handles) {
      Status status = pool.complete_reclaim_event(handle, event, true, true);
      if (!status.ok()) return status;
      const QwenKvCompletionEvent scrub{event.handle + 1,
                                        event.generation + handle.slot + 1};
      auto work = pool.begin_next_scrub(scrub);
      if (!work.ok()) return work.status();
      status = pool.complete_scrub(work->slot, scrub, work->bytes, true, true);
      if (!status.ok()) return status;
    }
    return status;
  }
  Status status = Status::Ok();
  int calls = 0;
  QwenKvCompletionEvent seen_event{};
};

Result<QwenKvSlotPool> ready_pool() {
  auto pool = QwenKvSlotPool::Create(
      8, 8 * QwenKvSlotPool::kSlotPayloadBytes,
      8 * sizeof(QwenKvSlotState));
  if (!pool.ok()) return pool.status();
  Status ready = pool->complete_startup_sanitize(
      8 * QwenKvSlotPool::kSlotPayloadBytes,
      8 * sizeof(QwenKvSlotState), true);
  if (!ready.ok()) return ready;
  return pool;
}

TEST(QwenBf16RequestRunnerTest, GeneratesAndReturnsEveryKvCredit) {
  auto pool = ready_pool().value();
  Backend backend;
  Completion completion;
  Recycler recycler;
  auto runner = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2, 3);
  ASSERT_TRUE(runner.ok());

  const std::int64_t prompt[]{1, 3, 5};
  auto generated = runner->generate(prompt, 4);
  ASSERT_TRUE(generated.ok());
  EXPECT_EQ(*generated, (std::vector<std::int64_t>{7, 2}));
  EXPECT_EQ(backend.calls, (std::vector<std::size_t>{3, 1}));
  EXPECT_EQ(recycler.calls, 1);
  EXPECT_EQ(recycler.seen_event.handle, 31U);
  EXPECT_EQ(recycler.seen_event.generation, 37U);
  EXPECT_EQ(pool.clean_credits(), 8U);
  EXPECT_EQ(runner->state(), QwenBf16RequestRunnerState::kReady);
}

TEST(QwenBf16RequestRunnerTest, BackendFailurePermanentlyPoisonsRunner) {
  auto pool = ready_pool().value();
  Backend backend;
  backend.status = Status::Internal("kernel failed");
  Completion completion;
  Recycler recycler;
  auto runner = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2);
  ASSERT_TRUE(runner.ok());
  const std::int64_t prompt[]{1};
  EXPECT_FALSE(runner->generate(prompt, 1).ok());
  EXPECT_EQ(runner->state(), QwenBf16RequestRunnerState::kPoisoned);
  EXPECT_EQ(recycler.calls, 0);
  EXPECT_FALSE(runner->generate(prompt, 1).ok());
}

TEST(QwenBf16RequestRunnerTest, RejectsBadRequestWithoutPoisoning) {
  auto pool = ready_pool().value();
  Backend backend;
  Completion completion;
  Recycler recycler;
  auto runner = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2).value();
  EXPECT_FALSE(runner.generate({}, 1).ok());
  const std::int64_t invalid[]{QwenBf16SequenceSession::kVocabularySize};
  EXPECT_FALSE(runner.generate(invalid, 1).ok());
  EXPECT_EQ(runner.state(), QwenBf16RequestRunnerState::kReady);
}

TEST(QwenBf16RequestRunnerTest, ChunkedPrefillPublishesOnlyFinalPrediction) {
  auto pool = ready_pool().value();
  Backend backend;
  Completion completion;
  Recycler recycler;
  auto runner = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2, 2).value();
  const std::int64_t prompt[]{1, 3, 5, 7, 9};
  auto generated = runner.generate(prompt, 1);
  ASSERT_TRUE(generated.ok());
  EXPECT_EQ(*generated, (std::vector<std::int64_t>{2}));
  EXPECT_EQ(backend.calls, (std::vector<std::size_t>{2, 2, 1}));
  EXPECT_EQ(pool.clean_credits(), 8U);
}

TEST(QwenBf16RequestRunnerTest, TailPrefillUsesOnlyFrozenDecodeShape) {
  auto pool = ready_pool().value();
  Backend backend;
  Completion completion;
  Recycler recycler;
  auto runner = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2, 4).value();
  const std::int64_t prompt[]{1, 3, 5, 7, 9, 11};

  auto generated = runner.generate(prompt, 1);

  ASSERT_TRUE(generated.ok());
  EXPECT_EQ(backend.calls, (std::vector<std::size_t>{4, 1, 1}));
  EXPECT_EQ(pool.clean_credits(), 8U);
}

TEST(QwenBf16RequestRunnerTest, GenerationIdentitySurvivesRunnerRecreation) {
  auto pool = ready_pool().value();
  Backend backend;
  Completion completion;
  Recycler recycler;
  const std::int64_t prompt[]{1};

  auto first = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2).value();
  ASSERT_TRUE(first.generate(prompt, 1).ok());
  auto second = QwenBf16RequestRunner::Create(
      pool, backend, completion, recycler, 2).value();
  ASSERT_TRUE(second.generate(prompt, 1).ok());

  EXPECT_EQ(backend.sequence_generations,
            (std::vector<std::uint32_t>{1, 2}));
  EXPECT_EQ(pool.clean_credits(), 8U);
}

}  // namespace
}  // namespace pih
