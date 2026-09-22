#pragma once

#include <array>
#include <span>

#include "pih/core/canonical_hash.h"

namespace pih {

enum class DeepSeekDsparkPrefixProofStepKind : std::uint8_t {
  kSiblingLeft = 1,
  kSiblingRight = 2,
  kPromote = 3,
};
struct DeepSeekDsparkPrefixProofStep final {
  DeepSeekDsparkPrefixProofStepKind kind{};
  Sha256Digest sibling;
};
struct DeepSeekDsparkPrefixProof final {
  std::uint32_t rank = 0;
  std::uint32_t world_size = 0;
  Sha256Digest local_state_hash;
  std::array<DeepSeekDsparkPrefixProofStep, 2> steps{};
  std::uint32_t step_count = 0;
};
struct DeepSeekDsparkPrefixCommitment final {
  Sha256Digest root;
  std::array<DeepSeekDsparkPrefixProof, 4> proofs{};
  std::uint32_t world_size = 0;
};

Result<DeepSeekDsparkPrefixCommitment> compile_deepseek_dspark_prefix_commitment(
    std::uint64_t plan_sequence, std::uint64_t old_generation,
    std::uint64_t new_generation,
    std::span<const Sha256Digest> rank_state_hashes);
Status verify_deepseek_dspark_prefix_proof(
    std::uint64_t plan_sequence, std::uint64_t old_generation,
    std::uint64_t new_generation, const DeepSeekDsparkPrefixProof& proof,
    const Sha256Digest& expected_root);

}  // namespace pih
