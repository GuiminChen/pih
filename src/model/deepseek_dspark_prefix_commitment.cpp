#include "pih/model/deepseek_dspark_prefix_commitment.h"

namespace pih { namespace {
Result<Sha256Digest> leaf(std::uint64_t plan, std::uint64_t old_generation,
                          std::uint64_t new_generation, std::uint32_t rank,
                          const Sha256Digest& state_hash) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-dspark-prefix-leaf:v1", 5);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_u64(1, plan);
  if (status.ok()) status = hash->add_u64(2, old_generation);
  if (status.ok()) status = hash->add_u64(3, new_generation);
  if (status.ok()) status = hash->add_u32(4, rank);
  if (status.ok()) status = hash->add_hash(5, state_hash);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}
Result<Sha256Digest> node(std::uint32_t level, const Sha256Digest& left,
                          const Sha256Digest& right) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-dspark-prefix-node:v1", 3);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_u32(1, level);
  if (status.ok()) status = hash->add_hash(2, left);
  if (status.ok()) status = hash->add_hash(3, right);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}
Result<Sha256Digest> promote(std::uint32_t level,
                             const Sha256Digest& child) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-dspark-prefix-promote:v1", 2);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_u32(1, level);
  if (status.ok()) status = hash->add_hash(2, child);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}
Result<Sha256Digest> wrap(std::uint32_t world_size,
                          const Sha256Digest& tree_root) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-dspark-prefix-root:v1", 2);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_u32(1, world_size);
  if (status.ok()) status = hash->add_hash(2, tree_root);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}
void sibling(DeepSeekDsparkPrefixProof& proof,
             DeepSeekDsparkPrefixProofStepKind kind,
             const Sha256Digest& digest) {
  proof.steps[proof.step_count++] = {kind, digest};
}
void promotion(DeepSeekDsparkPrefixProof& proof) {
  proof.steps[proof.step_count++] = {
      DeepSeekDsparkPrefixProofStepKind::kPromote, {}};
}
Result<DeepSeekDsparkPrefixProofStepKind> expected_step(
    std::uint32_t world, std::uint32_t rank, std::uint32_t level) {
  using Kind = DeepSeekDsparkPrefixProofStepKind;
  if (world == 2 && level == 0)
    return rank == 0 ? Kind::kSiblingRight : Kind::kSiblingLeft;
  if (world == 3 && level == 0) {
    if (rank == 0) return Kind::kSiblingRight;
    if (rank == 1) return Kind::kSiblingLeft;
    return Kind::kPromote;
  }
  if (world == 3 && level == 1)
    return rank < 2 ? Kind::kSiblingRight : Kind::kSiblingLeft;
  if (world == 4 && level == 0)
    return (rank & 1U) == 0 ? Kind::kSiblingRight : Kind::kSiblingLeft;
  if (world == 4 && level == 1)
    return rank < 2 ? Kind::kSiblingRight : Kind::kSiblingLeft;
  return Status::InvalidArgument(
      "DeepSeek DSpark prefix proof topology is invalid");
}
}  // namespace pih::<anonymous>

Result<DeepSeekDsparkPrefixCommitment> compile_deepseek_dspark_prefix_commitment(
    std::uint64_t plan, std::uint64_t old_generation,
    std::uint64_t new_generation,
    std::span<const Sha256Digest> state_hashes) {
  if (plan == 0 || old_generation == 0 ||
      new_generation != old_generation + 1 || state_hashes.empty() ||
      state_hashes.size() > 4) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefix commitment identity is invalid");
  }
  DeepSeekDsparkPrefixCommitment result;
  result.world_size = static_cast<std::uint32_t>(state_hashes.size());
  std::array<Sha256Digest, 4> leaves{};
  for (std::uint32_t rank = 0; rank < result.world_size; ++rank) {
    auto value = leaf(plan, old_generation, new_generation, rank,
                      state_hashes[rank]);
    if (!value.ok()) return value.status();
    leaves[rank] = *value;
    result.proofs[rank].rank = rank;
    result.proofs[rank].world_size = result.world_size;
    result.proofs[rank].local_state_hash = state_hashes[rank];
  }
  Sha256Digest tree = leaves[0];
  if (result.world_size >= 2) {
    auto first = node(0, leaves[0], leaves[1]);
    if (!first.ok()) return first.status();
    tree = *first;
    sibling(result.proofs[0],
            DeepSeekDsparkPrefixProofStepKind::kSiblingRight, leaves[1]);
    sibling(result.proofs[1],
            DeepSeekDsparkPrefixProofStepKind::kSiblingLeft, leaves[0]);
    if (result.world_size == 3) {
      auto odd = promote(0, leaves[2]);
      if (!odd.ok()) return odd.status();
      auto top = node(1, tree, *odd);
      if (!top.ok()) return top.status();
      sibling(result.proofs[0],
              DeepSeekDsparkPrefixProofStepKind::kSiblingRight, *odd);
      sibling(result.proofs[1],
              DeepSeekDsparkPrefixProofStepKind::kSiblingRight, *odd);
      promotion(result.proofs[2]);
      sibling(result.proofs[2],
              DeepSeekDsparkPrefixProofStepKind::kSiblingLeft, tree);
      tree = *top;
    } else if (result.world_size == 4) {
      auto second = node(0, leaves[2], leaves[3]);
      if (!second.ok()) return second.status();
      auto top = node(1, tree, *second);
      if (!top.ok()) return top.status();
      sibling(result.proofs[0],
              DeepSeekDsparkPrefixProofStepKind::kSiblingRight, *second);
      sibling(result.proofs[1],
              DeepSeekDsparkPrefixProofStepKind::kSiblingRight, *second);
      sibling(result.proofs[2],
              DeepSeekDsparkPrefixProofStepKind::kSiblingRight, leaves[3]);
      sibling(result.proofs[2],
              DeepSeekDsparkPrefixProofStepKind::kSiblingLeft, tree);
      sibling(result.proofs[3],
              DeepSeekDsparkPrefixProofStepKind::kSiblingLeft, leaves[2]);
      sibling(result.proofs[3],
              DeepSeekDsparkPrefixProofStepKind::kSiblingLeft, tree);
      tree = *top;
    }
  }
  auto root = wrap(result.world_size, tree);
  if (!root.ok()) return root.status();
  result.root = *root;
  return result;
}

Status verify_deepseek_dspark_prefix_proof(
    std::uint64_t plan, std::uint64_t old_generation,
    std::uint64_t new_generation, const DeepSeekDsparkPrefixProof& proof,
    const Sha256Digest& expected_root) {
  if (proof.world_size < 1 || proof.world_size > 4 ||
      proof.rank >= proof.world_size || proof.step_count > 2) {
    return Status::InvalidArgument("DeepSeek DSpark prefix proof is invalid");
  }
  const std::uint32_t required = proof.world_size == 1 ? 0 :
                                 proof.world_size == 2 ? 1 : 2;
  if (proof.step_count != required) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark prefix proof path length mismatched");
  }
  auto current = leaf(plan, old_generation, new_generation, proof.rank,
                      proof.local_state_hash);
  if (!current.ok()) return current.status();
  for (std::uint32_t level = 0; level < required; ++level) {
    const auto& step = proof.steps[level];
    auto expected = expected_step(proof.world_size, proof.rank, level);
    if (!expected.ok() || step.kind != *expected) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark prefix proof grammar mismatched");
    }
    Result<Sha256Digest> next = Status::Internal("unreachable proof step");
    if (step.kind == DeepSeekDsparkPrefixProofStepKind::kPromote) {
      next = promote(level, *current);
    } else if (step.kind ==
               DeepSeekDsparkPrefixProofStepKind::kSiblingLeft) {
      next = node(level, step.sibling, *current);
    } else {
      next = node(level, *current, step.sibling);
    }
    if (!next.ok()) return next.status();
    current = std::move(next);
  }
  auto root = wrap(proof.world_size, *current);
  if (!root.ok()) return root.status();
  return *root == expected_root
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark prefix proof root mismatched");
}

}  // namespace pih
