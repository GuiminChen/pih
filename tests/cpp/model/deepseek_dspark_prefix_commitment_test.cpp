#include "pih/model/deepseek_dspark_prefix_commitment.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {
Sha256Digest state_hash(std::uint32_t rank) {
  const std::string value = "rank-state-" + std::to_string(rank);
  return sha256(std::as_bytes(std::span(value))).value();
}
TEST(DeepSeekDsparkPrefixCommitmentTest,
     CompilesAndVerifiesEveryRankForPpOneThroughFour) {
  std::vector<Sha256Digest> hashes;
  for (std::uint32_t world = 1; world <= 4; ++world) {
    hashes.push_back(state_hash(world - 1));
    auto commitment = compile_deepseek_dspark_prefix_commitment(
        11, 5, 6, hashes);
    ASSERT_TRUE(commitment.ok()) << commitment.status().message();
    EXPECT_EQ(commitment->world_size, world);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      EXPECT_TRUE(verify_deepseek_dspark_prefix_proof(
          11, 5, 6, commitment->proofs[rank], commitment->root).ok());
    }
  }
}
TEST(DeepSeekDsparkPrefixCommitmentTest,
     PpThreeUsesExplicitOddLeafPromotion) {
  std::array<Sha256Digest, 3> hashes{
      state_hash(0), state_hash(1), state_hash(2)};
  auto commitment = compile_deepseek_dspark_prefix_commitment(
      11, 5, 6, hashes).value();
  ASSERT_EQ(commitment.proofs[2].step_count, 2U);
  EXPECT_EQ(commitment.proofs[2].steps[0].kind,
            DeepSeekDsparkPrefixProofStepKind::kPromote);
  EXPECT_EQ(commitment.proofs[2].steps[1].kind,
            DeepSeekDsparkPrefixProofStepKind::kSiblingLeft);
}
TEST(DeepSeekDsparkPrefixCommitmentTest,
     RejectsMutatedIdentityLeafPathAndWorldSize) {
  std::array<Sha256Digest, 4> hashes{
      state_hash(0), state_hash(1), state_hash(2), state_hash(3)};
  auto commitment = compile_deepseek_dspark_prefix_commitment(
      11, 5, 6, hashes).value();
  auto proof = commitment.proofs[2];
  EXPECT_FALSE(verify_deepseek_dspark_prefix_proof(
      12, 5, 6, proof, commitment.root).ok());
  proof.local_state_hash = state_hash(9);
  EXPECT_FALSE(verify_deepseek_dspark_prefix_proof(
      11, 5, 6, proof, commitment.root).ok());
  proof = commitment.proofs[2];
  proof.steps[1].kind = DeepSeekDsparkPrefixProofStepKind::kSiblingRight;
  EXPECT_FALSE(verify_deepseek_dspark_prefix_proof(
      11, 5, 6, proof, commitment.root).ok());
  proof = commitment.proofs[2]; proof.world_size = 3;
  EXPECT_FALSE(verify_deepseek_dspark_prefix_proof(
      11, 5, 6, proof, commitment.root).ok());
}
}}  // namespace pih::<anonymous>
