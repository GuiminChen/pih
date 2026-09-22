#include "pih/model/deepseek_dspark_greedy_verify.h"

namespace pih {
Result<DeepSeekDsparkGreedyVerifyResult> deepseek_dspark_greedy_verify(
    std::span<const std::uint32_t> drafts,
    std::span<const std::uint32_t> targets, std::uint32_t vocab_size) {
  if (drafts.empty() || drafts.size() > 5 || drafts.size() != targets.size() ||
      vocab_size != 129280) {
    return Status::InvalidArgument(
        "DeepSeek DSpark greedy verify shape is invalid");
  }
  for (std::size_t index = 0; index < drafts.size(); ++index) {
    if (drafts[index] >= vocab_size || targets[index] >= vocab_size)
      return Status::InvalidArgument(
          "DeepSeek DSpark greedy verify token is invalid");
  }
  DeepSeekDsparkGreedyVerifyResult result;
  for (std::size_t index = 0; index < drafts.size(); ++index) {
    if (drafts[index] == targets[index]) {
      result.retained_tokens[result.retained_record_count++] = drafts[index];
      ++result.accepted_draft_count;
      continue;
    }
    result.retained_tokens[result.retained_record_count++] = targets[index];
    result.mismatch = true;
    return result;
  }
  return result;
}
}  // namespace pih
