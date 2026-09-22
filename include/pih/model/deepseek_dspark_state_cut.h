#pragma once

#include <cstdint>
#include <optional>

#include "pih/core/sha256.h"
#include "pih/model/deepseek_dspark_greedy_verify.h"

namespace pih {

struct DeepSeekDsparkStateCutDecision final {
  std::uint64_t plan_sequence = 0;
  std::uint64_t old_generation = 0;
  std::uint64_t new_generation = 0;
  std::uint32_t retained_record_count = 0;
  std::uint32_t processed_delta = 0;
  std::optional<std::uint32_t> pending_input_token;
  bool terminal_drain = false;
  Sha256Digest per_rank_prefix_hash_merkle_root;
};

Result<DeepSeekDsparkStateCutDecision> compile_deepseek_dspark_state_cut(
    std::uint64_t plan_sequence, std::uint64_t old_generation,
    const DeepSeekDsparkGreedyVerifyResult& verified,
    std::uint32_t retained_record_count, bool terminal,
    const Sha256Digest& per_rank_prefix_hash_merkle_root);

}  // namespace pih
