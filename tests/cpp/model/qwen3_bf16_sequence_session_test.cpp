#include "pih/model/qwen3_bf16_sequence_session.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

QwenKvSlotPool ready_pool(std::uint32_t slots) {
  auto pool = QwenKvSlotPool::Create(
                  slots, slots * QwenKvSlotPool::kSlotPayloadBytes,
                  slots * sizeof(QwenKvSlotState))
                  .value();
  EXPECT_TRUE(pool.complete_startup_sanitize(
                      slots * QwenKvSlotPool::kSlotPayloadBytes,
                      slots * sizeof(QwenKvSlotState), true)
                  .ok());
  return pool;
}

class Backend final : public QwenBf16SequenceBackend {
 public:
  Result<std::int64_t> execute_and_read_token(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& table,
      const QwenKvAppendPlan& append) override {
    calls.emplace_back(tokens.begin(), tokens.end());
    positions.push_back(first_position);
    visible_before.push_back(table.visible_handles().size());
    visible_after.push_back(append.visible_handle_count_after_commit);
    if (fail) return Status::Internal("injected sequence failure");
    return next_token++;
  }

  bool fail = false;
  std::int64_t next_token = 20;
  std::vector<std::vector<std::int64_t>> calls;
  std::vector<std::uint64_t> positions;
  std::vector<std::size_t> visible_before;
  std::vector<std::uint32_t> visible_after;
};

TEST(QwenBf16SequenceSessionTest, CommitsPrefillAndDecodeOnlyAfterBackendSuccess) {
  auto pool = ready_pool(3);
  Backend backend;
  auto session = QwenBf16SequenceSession::Admit(pool, 4, 7, 33, backend);
  ASSERT_TRUE(session.ok()) << session.status().message();
  const std::int64_t prompt[] = {1, 2, 3};
  auto first = session->execute(prompt, 0);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(*first, 20);
  EXPECT_EQ(session->committed_tokens(), 3);
  const std::int64_t decoded[] = {20};
  auto second = session->execute(decoded, 3);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(session->committed_tokens(), 4);
  EXPECT_EQ(backend.visible_before, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(backend.visible_after, (std::vector<std::uint32_t>{1, 1}));
  EXPECT_EQ(pool.slot(0).valid_tokens, 4);
}

TEST(QwenBf16SequenceSessionTest, BackendFailureRollsBackAndPermanentlyPoisons) {
  auto pool = ready_pool(2);
  Backend backend;
  backend.fail = true;
  auto session = QwenBf16SequenceSession::Admit(pool, 3, 2, 17, backend);
  ASSERT_TRUE(session.ok());
  const std::int64_t prompt[] = {1, 2};
  EXPECT_FALSE(session->execute(prompt, 0).ok());
  EXPECT_EQ(session->committed_tokens(), 0);
  EXPECT_EQ(pool.slot(0).valid_tokens, 0);
  EXPECT_EQ(session->state(), QwenBf16SequenceSessionState::kPoisoned);
  EXPECT_FALSE(session->execute(prompt, 0).ok());
  EXPECT_EQ(backend.calls.size(), 1);
  EXPECT_TRUE(session->release({8, 1}).ok());
}

TEST(QwenBf16SequenceSessionTest, RejectsPositionDriftBeforeBackendSubmission) {
  auto pool = ready_pool(2);
  Backend backend;
  auto session = QwenBf16SequenceSession::Admit(pool, 3, 2, 17, backend);
  ASSERT_TRUE(session.ok());
  const std::int64_t prompt[] = {1, 2};
  EXPECT_FALSE(session->execute(prompt, 1).ok());
  EXPECT_TRUE(backend.calls.empty());
  EXPECT_EQ(session->state(), QwenBf16SequenceSessionState::kPoisoned);
}

TEST(QwenBf16SequenceSessionTest, InvalidBackendTokenCannotCommitKvPrefix) {
  auto pool = ready_pool(2);
  Backend backend;
  backend.next_token = QwenBf16SequenceSession::kVocabularySize;
  auto session = QwenBf16SequenceSession::Admit(pool, 3, 2, 17, backend);
  ASSERT_TRUE(session.ok());
  const std::int64_t prompt[] = {1, 2};
  EXPECT_FALSE(session->execute(prompt, 0).ok());
  EXPECT_EQ(session->committed_tokens(), 0);
  EXPECT_EQ(pool.slot(0).valid_tokens, 0);
  EXPECT_EQ(session->state(), QwenBf16SequenceSessionState::kPoisoned);
}

}  // namespace
}  // namespace pih
