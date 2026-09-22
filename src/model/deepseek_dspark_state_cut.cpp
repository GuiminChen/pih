#include "pih/model/deepseek_dspark_state_cut.h"

#include <limits>

namespace pih { namespace {
bool empty(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) if (value != std::byte{0}) return false;
  return true;
}
}  // namespace pih::<anonymous>

Result<DeepSeekDsparkStateCutDecision> compile_deepseek_dspark_state_cut(
    std::uint64_t plan_sequence, std::uint64_t old_generation,
    const DeepSeekDsparkGreedyVerifyResult& verified,
    std::uint32_t retained_count, bool terminal,
    const Sha256Digest& prefix_root) {
  if (plan_sequence == 0 || old_generation == 0 ||
      old_generation == std::numeric_limits<std::uint64_t>::max() ||
      verified.retained_record_count == 0 ||
      verified.retained_record_count > 5 || retained_count == 0 ||
      retained_count > verified.retained_record_count || empty(prefix_root)) {
    return Status::InvalidArgument(
        "DeepSeek DSpark state cut input is invalid");
  }
  DeepSeekDsparkStateCutDecision result;
  result.plan_sequence = plan_sequence;
  result.old_generation = old_generation;
  result.new_generation = old_generation + 1;
  result.retained_record_count = retained_count;
  result.processed_delta = retained_count;
  result.terminal_drain = terminal;
  if (!terminal)
    result.pending_input_token = verified.retained_tokens[retained_count - 1];
  result.per_rank_prefix_hash_merkle_root = prefix_root;
  return result;
}

}  // namespace pih
