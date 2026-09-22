#pragma once

#include <cstdint>
#include <span>

#include "pih/model/deepseek_dspark_state_cut.h"

namespace pih {

struct DeepSeekDsparkStateCutAck final {
  std::uint32_t rank = 0;
  std::uint64_t plan_sequence = 0;
  std::uint64_t old_generation = 0;
  std::uint64_t new_generation = 0;
  std::uint32_t retained_record_count = 0;
  std::uint32_t processed_delta = 0;
  bool terminal_drain = false;
  bool prefix_published = false;
  bool cuda_complete = false;
  bool nccl_complete = false;
  Sha256Digest per_rank_prefix_hash_merkle_root;
};

struct DeepSeekDsparkStateCutQuorum final {
  std::uint32_t world_size = 0;
  DeepSeekDsparkStateCutDecision decision;
};

Result<DeepSeekDsparkStateCutQuorum> verify_deepseek_dspark_state_cut_acks(
    const DeepSeekDsparkStateCutDecision& decision,
    std::uint32_t world_size,
    std::span<const DeepSeekDsparkStateCutAck> acknowledgements,
    bool engine_poisoned);

}  // namespace pih
