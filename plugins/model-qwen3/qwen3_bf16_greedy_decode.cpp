#include "pih/model/qwen3_bf16_greedy_decode.h"

#include <limits>

namespace pih {

Result<QwenBf16GreedyDecode> QwenBf16GreedyDecode::Create(
    std::span<const std::int64_t> prompt, std::uint64_t maximum_new_tokens,
    std::int64_t eos_token_id) {
  if (prompt.empty() || maximum_new_tokens == 0 ||
      prompt.size() >= kMaximumPositions ||
      maximum_new_tokens > kMaximumPositions - prompt.size()) {
    return Status::InvalidArgument(
        "Qwen greedy decode request exceeds its bounded context");
  }
  if (eos_token_id < 0) {
    return Status::InvalidArgument("Qwen greedy decode EOS token is invalid");
  }
  for (const std::int64_t token : prompt) {
    if (token < 0) {
      return Status::InvalidArgument(
          "Qwen greedy decode prompt contains an invalid token");
    }
  }
  return QwenBf16GreedyDecode(
      std::vector<std::int64_t>(prompt.begin(), prompt.end()),
      maximum_new_tokens, eos_token_id);
}

Status QwenBf16GreedyDecode::run_next(QwenBf16GreedyStepDriver& driver) {
  if (state_ == QwenBf16GreedyDecodeState::kCompleted ||
      state_ == QwenBf16GreedyDecodeState::kPoisoned) {
    return Status::FailedPrecondition(
        "Qwen greedy decode has no runnable step");
  }
  state_ = QwenBf16GreedyDecodeState::kRunning;
  const std::int64_t previous =
      generated_.empty() ? 0 : generated_.back();
  const std::span<const std::int64_t> input = generated_.empty()
      ? std::span<const std::int64_t>(prompt_)
      : std::span<const std::int64_t>(&previous, 1);
  const std::uint64_t first_position =
      generated_.empty() ? 0 : prompt_.size() + generated_.size() - 1;
  auto token = driver.execute(input, first_position);
  if (!token.ok()) {
    state_ = QwenBf16GreedyDecodeState::kPoisoned;
    return token.status();
  }
  if (*token < 0) {
    state_ = QwenBf16GreedyDecodeState::kPoisoned;
    return Status::Internal("Qwen greedy decode produced an invalid token");
  }
  generated_.push_back(*token);
  if (*token == eos_token_id_ || generated_.size() == maximum_new_tokens_) {
    state_ = QwenBf16GreedyDecodeState::kCompleted;
  }
  return Status::Ok();
}

}  // namespace pih
