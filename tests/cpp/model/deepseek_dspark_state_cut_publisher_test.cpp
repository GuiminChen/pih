#include "pih/model/deepseek_dspark_state_cut_publisher.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {
class Operations final : public DeepSeekDsparkStateCutPublishOperations {
 public:
  Status validate_engine_healthy() override {
    calls.push_back("health");
    return fail_health_at == health_calls++ ? Status::Internal("poison")
                                            : Status::Ok();
  }
  Status publish_ledger_and_output(const DeepSeekDsparkStateCutDecision&) override {
    calls.push_back("ledger"); return fail_ledger ? Status::Internal("ledger")
                                                  : Status::Ok();
  }
  std::vector<std::string> calls;
  int fail_health_at = -1;
  int health_calls = 0;
  bool fail_ledger = false;
};
DeepSeekDsparkStateCutDecision decision(bool terminal = false) {
  Sha256Digest root; root.bytes[0] = std::byte{1};
  return {9, 41, 42, 3, 3,
          terminal ? std::optional<std::uint32_t>{}
                   : std::optional<std::uint32_t>{17},
          terminal, root};
}
std::vector<DeepSeekDsparkStateCutAck> acknowledgements(
    const DeepSeekDsparkStateCutDecision& cut) {
  std::vector<DeepSeekDsparkStateCutAck> result;
  for (std::uint32_t rank = 0; rank < 2; ++rank) {
    result.push_back({rank, cut.plan_sequence, cut.old_generation,
                      cut.new_generation, cut.retained_record_count,
                      cut.processed_delta, cut.terminal_drain, true, true, true,
                      cut.per_rank_prefix_hash_merkle_root});
  }
  return result;
}
TEST(DeepSeekDsparkStateCutPublisherTest,
     RechecksHealthBetweenPrefixAndLedgerPublication) {
  Operations operations;
  auto publisher = DeepSeekDsparkStateCutPublisher::Create(operations).value();
  auto cut = decision(); auto acks = acknowledgements(cut);
  ASSERT_TRUE(publisher.publish(cut, 2, acks, false).ok());
  EXPECT_EQ(operations.calls,
            std::vector<std::string>({"health", "ledger"}));
  EXPECT_TRUE(publisher.published());
  EXPECT_FALSE(publisher.publish(cut, 2, acks, false).ok());
}
TEST(DeepSeekDsparkStateCutPublisherTest,
     TerminalDecisionPublishesLedgerAfterRankDrainAcknowledgements) {
  Operations operations;
  auto publisher = DeepSeekDsparkStateCutPublisher::Create(operations).value();
  auto cut = decision(true); auto acks = acknowledgements(cut);
  ASSERT_TRUE(publisher.publish(cut, 2, acks, false).ok());
  EXPECT_EQ(operations.calls,
            std::vector<std::string>({"health", "ledger"}));
}
TEST(DeepSeekDsparkStateCutPublisherTest,
     LedgerFailurePoisonsAndCannotRetry) {
  Operations operations; operations.fail_ledger = true;
  auto publisher = DeepSeekDsparkStateCutPublisher::Create(operations).value();
  auto cut = decision(); auto acks = acknowledgements(cut);
  EXPECT_FALSE(publisher.publish(cut, 2, acks, false).ok());
  EXPECT_TRUE(publisher.poisoned());
  operations.fail_ledger = false;
  EXPECT_FALSE(publisher.publish(cut, 2, acks, false).ok());
  EXPECT_EQ(operations.calls.back(), "ledger");
}
TEST(DeepSeekDsparkStateCutPublisherTest,
     RejectsUnverifiedAcknowledgementsBeforeAnyMutation) {
  Operations operations;
  auto publisher = DeepSeekDsparkStateCutPublisher::Create(operations).value();
  auto cut = decision(); auto acks = acknowledgements(cut);
  acks.pop_back();
  EXPECT_FALSE(publisher.publish(cut, 2, acks, false).ok());
  EXPECT_TRUE(operations.calls.empty());
  EXPECT_FALSE(publisher.poisoned());
}
}}  // namespace pih::<anonymous>
