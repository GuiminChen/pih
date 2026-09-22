#include "pih/model/qwen3_bf16_execution_cursor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

Qwen3Config official_config() {
  return Qwen3Config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                     1'000'000.0, 0.000001, 151643, 151645};
}

class RecordingDriver final : public QwenBf16ExecutionDriver {
 public:
  Status Execute(const QwenBf16ExecutionStep& step,
                 const QwenBf16ExecutionRequest& request) override {
    steps.push_back(step);
    requests.push_back(request);
    if (steps.size() == fail_at) {
      return Status::Unavailable("injected backend failure");
    }
    return Status::Ok();
  }

  std::size_t fail_at = 0;
  std::vector<QwenBf16ExecutionStep> steps;
  std::vector<QwenBf16ExecutionRequest> requests;
};

TEST(QwenBf16ExecutionCursorTest, RunsExactScheduleAndPublishesCompletion) {
  auto cursor = QwenBf16ExecutionCursor::Create(official_config());
  ASSERT_TRUE(cursor.ok());
  EXPECT_EQ(cursor->state(), QwenBf16ExecutionState::kReady);
  ASSERT_TRUE(cursor->Begin(7, 31).ok());
  EXPECT_EQ(cursor->generation(), 1);

  RecordingDriver driver;
  while (cursor->state() == QwenBf16ExecutionState::kRunning) {
    ASSERT_TRUE(cursor->RunNext(driver).ok());
  }
  EXPECT_EQ(cursor->state(), QwenBf16ExecutionState::kCompleted);
  ASSERT_EQ(driver.steps.size(), QwenBf16ExecutionSchedule::kStepCount);
  EXPECT_EQ(driver.steps.front().operation, QwenBf16ExecutionOp::kEmbedding);
  EXPECT_EQ(driver.steps.back().operation, QwenBf16ExecutionOp::kGreedyArgmax);
  for (const auto& request : driver.requests) {
    EXPECT_EQ(request.tokens, 7);
    EXPECT_EQ(request.first_position, 31);
    EXPECT_EQ(request.active_logit_rows, 1);
    EXPECT_EQ(request.generation, 1);
  }

  ASSERT_TRUE(cursor->Begin(1, 39).ok());
  EXPECT_EQ(cursor->generation(), 2);
}

TEST(QwenBf16ExecutionCursorTest, BackendFailurePermanentlyPoisonsCursor) {
  auto cursor = QwenBf16ExecutionCursor::Create(official_config());
  ASSERT_TRUE(cursor.ok());
  ASSERT_TRUE(cursor->Begin(1, 0).ok());
  RecordingDriver driver;
  driver.fail_at = 3;
  const auto failure = cursor->RunNext(driver);
  ASSERT_TRUE(failure.ok());
  ASSERT_TRUE(cursor->RunNext(driver).ok());
  const auto injected = cursor->RunNext(driver);
  EXPECT_FALSE(injected.ok());
  EXPECT_EQ(injected.code(), StatusCode::kUnavailable);
  EXPECT_EQ(cursor->state(), QwenBf16ExecutionState::kPoisoned);
  EXPECT_FALSE(cursor->RunNext(driver).ok());
  EXPECT_FALSE(cursor->Begin(1, 0).ok());
  EXPECT_EQ(driver.steps.size(), 3);
}

TEST(QwenBf16ExecutionCursorTest, EnforcesRequestAndLifecycleBounds) {
  auto cursor = QwenBf16ExecutionCursor::Create(official_config());
  ASSERT_TRUE(cursor.ok());
  RecordingDriver driver;
  EXPECT_FALSE(cursor->RunNext(driver).ok());
  EXPECT_FALSE(cursor->Begin(0, 0).ok());
  EXPECT_FALSE(cursor->Begin(4097, 0).ok());
  EXPECT_FALSE(cursor->Begin(2, 40959).ok());
  ASSERT_TRUE(cursor->Begin(1, 40959).ok());
  EXPECT_FALSE(cursor->Begin(1, 0).ok());
}

}  // namespace
}  // namespace pih
