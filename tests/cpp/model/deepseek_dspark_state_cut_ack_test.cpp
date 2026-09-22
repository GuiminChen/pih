#include "pih/model/deepseek_dspark_state_cut_ack.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace pih { namespace {
Sha256Digest root() { Sha256Digest v; v.bytes[0]=std::byte{7}; return v; }
DeepSeekDsparkStateCutDecision decision() {
  return {9, 41, 42, 3, 3, 17, false, root()};
}
DeepSeekDsparkStateCutAck ack(std::uint32_t rank) {
  return {rank, 9, 41, 42, 3, 3, false, true, true, true, root()};
}
TEST(DeepSeekDsparkStateCutAckTest, AcceptsExactQuorumForWorldSizesOneToFour) {
  for (std::uint32_t world = 1; world <= 4; ++world) {
    std::vector<DeepSeekDsparkStateCutAck> acks;
    for (std::uint32_t rank = 0; rank < world; ++rank) acks.push_back(ack(rank));
    auto quorum = verify_deepseek_dspark_state_cut_acks(
        decision(), world, acks, false);
    ASSERT_TRUE(quorum.ok());
    EXPECT_EQ(quorum->world_size, world);
  }
}
TEST(DeepSeekDsparkStateCutAckTest, RejectsMissingDuplicateOrDriftedRank) {
  std::array acks{ack(0), ack(1)};
  EXPECT_FALSE(verify_deepseek_dspark_state_cut_acks(
      decision(), 3, acks, false).ok());
  acks[1].rank = 0;
  EXPECT_FALSE(verify_deepseek_dspark_state_cut_acks(
      decision(), 2, acks, false).ok());
  acks[1] = ack(1); acks[1].processed_delta = 2;
  EXPECT_FALSE(verify_deepseek_dspark_state_cut_acks(
      decision(), 2, acks, false).ok());
}
TEST(DeepSeekDsparkStateCutAckTest, RejectsIncompleteTransportAndPoison) {
  std::array acks{ack(0)};
  acks[0].cuda_complete = false;
  EXPECT_FALSE(verify_deepseek_dspark_state_cut_acks(
      decision(), 1, acks, false).ok());
  acks[0] = ack(0); acks[0].nccl_complete = false;
  EXPECT_FALSE(verify_deepseek_dspark_state_cut_acks(
      decision(), 1, acks, false).ok());
  acks[0] = ack(0);
  EXPECT_FALSE(verify_deepseek_dspark_state_cut_acks(
      decision(), 1, acks, true).ok());
}
}}  // namespace pih::<anonymous>
