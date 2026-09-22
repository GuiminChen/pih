#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_token_ledger.h"

namespace pih {
namespace {

Result<QwenSemanticOutcome> outcome(QwenSemanticTokenLedger& ledger) {
  QwenSemanticOutcomeRecorder recorder;
  const std::array payload{std::byte{7}};
  Status status = recorder.record_dispatch(payload);
  if (status.ok()) status = recorder.record_final_logits(payload);
  if (status.ok()) status = recorder.record_kv_state(payload);
  if (status.ok()) status = recorder.observe_capacity(100, 20);
  if (status.ok()) status = ledger.seal(recorder);
  if (!status.ok()) return status;
  return recorder.seal();
}

TEST(QwenSemanticTokenLedgerTest, FreezesCommittedChunksAndDecisions) {
  QwenSemanticTokenLedger ledger;
  const std::array<std::int64_t, 1> first{11};
  const std::array<std::int64_t, 2> second{12, 13};
  ASSERT_TRUE(ledger.commit(0, first, 0, 101).ok());
  ASSERT_TRUE(ledger.commit(1, second, 2, 102).ok());
  EXPECT_EQ(ledger.committed_tokens(), 3);
  EXPECT_EQ(ledger.decision_count(), 2);
  auto result = outcome(ledger);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(ledger.state(), QwenSemanticTokenLedgerState::kSealed);
}

TEST(QwenSemanticTokenLedgerTest, BoundariesAndSamplesAffectSeparateRoots) {
  const std::array<std::int64_t, 3> all{11, 12, 13};
  const std::array<std::int64_t, 1> first{11};
  const std::array<std::int64_t, 2> second{12, 13};

  QwenSemanticTokenLedger one_chunk;
  ASSERT_TRUE(one_chunk.commit(0, all, 2, 102).ok());
  auto one = outcome(one_chunk);
  ASSERT_TRUE(one.ok());

  QwenSemanticTokenLedger two_chunks;
  ASSERT_TRUE(two_chunks.commit(0, first, 0, 101).ok());
  ASSERT_TRUE(two_chunks.commit(1, second, 2, 102).ok());
  auto two = outcome(two_chunks);
  ASSERT_TRUE(two.ok());
  EXPECT_NE(one->accepted_token_ledger_root,
            two->accepted_token_ledger_root);
  EXPECT_NE(one->greedy_trajectory_root, two->greedy_trajectory_root);

  QwenSemanticTokenLedger sampled_drift;
  ASSERT_TRUE(sampled_drift.commit(0, first, 0, 99).ok());
  ASSERT_TRUE(sampled_drift.commit(1, second, 2, 102).ok());
  auto drift = outcome(sampled_drift);
  ASSERT_TRUE(drift.ok());
  EXPECT_EQ(two->accepted_token_ledger_root,
            drift->accepted_token_ledger_root);
  EXPECT_NE(two->greedy_trajectory_root, drift->greedy_trajectory_root);
}

TEST(QwenSemanticTokenLedgerTest, InvalidPositionOrTokenPoisons) {
  const std::array<std::int64_t, 1> token{11};
  QwenSemanticTokenLedger gap;
  EXPECT_FALSE(gap.commit(1, token, 1, 12).ok());
  EXPECT_EQ(gap.state(), QwenSemanticTokenLedgerState::kPoisoned);

  QwenSemanticTokenLedger invalid;
  const std::array<std::int64_t, 1> bad{151936};
  EXPECT_FALSE(invalid.commit(0, bad, 0, 12).ok());
  EXPECT_EQ(invalid.state(), QwenSemanticTokenLedgerState::kPoisoned);
}

}  // namespace
}  // namespace pih
