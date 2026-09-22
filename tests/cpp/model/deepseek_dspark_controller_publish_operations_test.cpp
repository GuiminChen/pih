#include "pih/model/deepseek_dspark_controller_publish_operations.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {
ControllerSequence verify_sequence() {
  auto sequence = ControllerSequence::Admit(7, 3, 41).value();
  EXPECT_TRUE(sequence.prepare(1, PackedTokenPhase::kPrefill, 3).ok());
  EXPECT_TRUE(sequence.commit(1).ok());
  EXPECT_TRUE(sequence.mark_in_flight(1).ok());
  EXPECT_TRUE(sequence.complete({1, 7, 3, 1, 42,
                                 PackedTokenPhase::kVerify, false}).ok());
  EXPECT_TRUE(sequence.prepare(9, PackedTokenPhase::kVerify, 5).ok());
  EXPECT_TRUE(sequence.commit(9).ok());
  EXPECT_TRUE(sequence.mark_in_flight(9).ok());
  return sequence;
}
DeepSeekDsparkStateCutDecision decision(bool terminal = false) {
  Sha256Digest root; root.bytes[0] = std::byte{1};
  return {9, 42, 43, 3, 3,
          terminal ? std::optional<std::uint32_t>{}
                   : std::optional<std::uint32_t>{17},
          terminal, root};
}
std::vector<DeepSeekDsparkStateCutAck> acknowledgement(
    const DeepSeekDsparkStateCutDecision& cut) {
  return {{0, cut.plan_sequence, cut.old_generation, cut.new_generation,
           cut.retained_record_count, cut.processed_delta, cut.terminal_drain,
           true, true, true, cut.per_rank_prefix_hash_merkle_root}};
}
TEST(DeepSeekDsparkControllerPublishOperationsTest,
     PublishesStateThenAdvancesControllerLedgerInvariant) {
  auto sequence = verify_sequence();
  auto operations = DeepSeekDsparkControllerPublishOperations::Create(
      sequence, 7, PackedTokenPhase::kVerify).value();
  auto publisher = DeepSeekDsparkStateCutPublisher::Create(operations).value();
  auto cut = decision();
  auto acks = acknowledgement(cut);
  ASSERT_TRUE(publisher.publish(cut, 1, acks, false).ok());
  EXPECT_EQ(sequence.model_processed_length(), 6U);
  EXPECT_EQ(sequence.accepted_completion_count(), 4U);
  EXPECT_EQ(sequence.state_generation(), 43U);
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kReadyDecode);
}
TEST(DeepSeekDsparkControllerPublishOperationsTest,
     TerminalCutDrainsStateAndMovesSequenceToDraining) {
  auto sequence = verify_sequence();
  auto operations = DeepSeekDsparkControllerPublishOperations::Create(
      sequence, 7, PackedTokenPhase::kDecode).value();
  auto publisher = DeepSeekDsparkStateCutPublisher::Create(operations).value();
  auto cut = decision(true);
  auto acks = acknowledgement(cut);
  ASSERT_TRUE(publisher.publish(cut, 1, acks, false).ok());
  EXPECT_EQ(sequence.state(), ControllerSequenceState::kDraining);
}
TEST(DeepSeekDsparkControllerPublishOperationsTest,
     RejectsNonVerifyOrStaleStateGenerationBeforeOwnerMutation) {
  auto sequence = ControllerSequence::Admit(7, 3, 41).value();
  EXPECT_FALSE(DeepSeekDsparkControllerPublishOperations::Create(
      sequence, 7, PackedTokenPhase::kVerify).ok());
  sequence = verify_sequence();
  auto operations = DeepSeekDsparkControllerPublishOperations::Create(
      sequence, 7, PackedTokenPhase::kVerify).value();
  auto stale = decision();
  stale.old_generation = 41;
  stale.new_generation = 42;
  EXPECT_FALSE(operations.publish_ledger_and_output(stale).ok());
}
}}  // namespace pih::<anonymous>
