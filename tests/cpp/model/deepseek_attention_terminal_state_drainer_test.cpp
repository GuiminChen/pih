#include "pih/model/deepseek_attention_terminal_state_drainer.h"

#include <gtest/gtest.h>

namespace pih { namespace {
DeepSeekDsparkStateCutDecision terminal_cut() {
  Sha256Digest root; root.bytes[0] = std::byte{9};
  return {11, 5, 6, 2, 2, std::nullopt, true, root};
}
struct DrainFixture final {
  DeepSeekStagePlan stage = DeepSeekPipelinePlan::Create(1, false).value().rank(0);
  std::uint64_t bytes =
      DeepSeekAttentionStateGeometry::StageLogicalBytes(stage, 4096).value();
  DeepSeekAttentionStatePool state_pool{bytes * 2};
  DeepSeekAttentionStateReservation reservation =
      state_pool.reserve(7, stage, 4096).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(4).value();
  DrainFixture() { EXPECT_TRUE(reservation.publish().ok()); }
};
TEST(DeepSeekAttentionTerminalStateDrainerTest,
     ReleasesLogicalCreditAndOnlyTargetSequencePages) {
  DrainFixture fixture;
  auto page4 = fixture.ratio4.reserve(7, 2, 0).value();
  auto page128 = fixture.ratio128.reserve(7, 3, 0).value();
  auto other4 = fixture.ratio4.reserve(8, 2, 0).value();
  auto other128 = fixture.ratio128.reserve(8, 3, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(page4).ok());
  ASSERT_TRUE(fixture.ratio128.publish(page128).ok());
  ASSERT_TRUE(fixture.ratio4.publish(other4).ok());
  ASSERT_TRUE(fixture.ratio128.publish(other128).ok());
  auto drainer = DeepSeekAttentionTerminalStateDrainer::Create(
      7, fixture.reservation, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(drainer.drain_committed_sequence_state(terminal_cut()).ok());
  EXPECT_TRUE(drainer.drained());
  EXPECT_EQ(fixture.reservation.state(),
            DeepSeekAttentionReservationState::kReleased);
  EXPECT_EQ(fixture.state_pool.owned_bytes(), 0U);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 1U);
  EXPECT_EQ(fixture.ratio128.published_pages(), 1U);
  EXPECT_TRUE(fixture.ratio4.release(other4).ok());
  EXPECT_TRUE(fixture.ratio128.release(other128).ok());
}
TEST(DeepSeekAttentionTerminalStateDrainerTest,
     PreflightFailureMutatesNoCommittedOwner) {
  DrainFixture fixture;
  auto published4 = fixture.ratio4.reserve(7, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(published4).ok());
  ASSERT_TRUE(fixture.ratio128.reserve(7, 3, 0).ok());
  auto drainer = DeepSeekAttentionTerminalStateDrainer::Create(
      7, fixture.reservation, fixture.ratio4, fixture.ratio128).value();
  EXPECT_FALSE(drainer.drain_committed_sequence_state(terminal_cut()).ok());
  EXPECT_EQ(fixture.reservation.state(),
            DeepSeekAttentionReservationState::kPublished);
  EXPECT_EQ(fixture.ratio4.published_pairs(), 1U);
  EXPECT_EQ(fixture.state_pool.owned_bytes(), fixture.bytes);
}
TEST(DeepSeekAttentionTerminalStateDrainerTest,
     RejectsNonterminalAndDuplicateDrain) {
  DrainFixture fixture;
  auto drainer = DeepSeekAttentionTerminalStateDrainer::Create(
      7, fixture.reservation, fixture.ratio4, fixture.ratio128).value();
  auto nonterminal = terminal_cut(); nonterminal.terminal_drain = false;
  EXPECT_FALSE(drainer.drain_committed_sequence_state(nonterminal).ok());
  ASSERT_TRUE(drainer.drain_committed_sequence_state(terminal_cut()).ok());
  EXPECT_FALSE(drainer.drain_committed_sequence_state(terminal_cut()).ok());
}
}}  // namespace pih::<anonymous>
