#include "pih/model/deepseek_dspark_state_cut_ack.h"

#include <array>

namespace pih {
Result<DeepSeekDsparkStateCutQuorum> verify_deepseek_dspark_state_cut_acks(
    const DeepSeekDsparkStateCutDecision& decision,
    std::uint32_t world_size,
    std::span<const DeepSeekDsparkStateCutAck> acknowledgements,
    bool engine_poisoned) {
  if (engine_poisoned || world_size < 1 || world_size > 4 ||
      acknowledgements.size() != world_size || decision.plan_sequence == 0 ||
      decision.old_generation == 0 ||
      decision.new_generation != decision.old_generation + 1 ||
      decision.retained_record_count == 0 ||
      decision.processed_delta != decision.retained_record_count) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark state-cut quorum is not verifiable");
  }
  std::array<bool, 4> seen{};
  for (const auto& ack : acknowledgements) {
    if (ack.rank >= world_size || seen[ack.rank] ||
        ack.plan_sequence != decision.plan_sequence ||
        ack.old_generation != decision.old_generation ||
        ack.new_generation != decision.new_generation ||
        ack.retained_record_count != decision.retained_record_count ||
        ack.processed_delta != decision.processed_delta ||
        ack.terminal_drain != decision.terminal_drain ||
        !ack.prefix_published || !ack.cuda_complete || !ack.nccl_complete ||
        ack.per_rank_prefix_hash_merkle_root !=
            decision.per_rank_prefix_hash_merkle_root) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark state-cut acknowledgement mismatched");
    }
    seen[ack.rank] = true;
  }
  return DeepSeekDsparkStateCutQuorum{world_size, decision};
}
}  // namespace pih
