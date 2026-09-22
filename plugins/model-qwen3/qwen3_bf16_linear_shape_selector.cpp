#include "pih/model/qwen3_bf16_linear_shape_selector.h"

namespace pih {

Result<QwenBf16LinearShapeSelector> QwenBf16LinearShapeSelector::Create(
    std::uint64_t prefill_tokens, std::uint64_t decode_tokens) {
  return Create(prefill_tokens, prefill_tokens, decode_tokens);
}

Result<QwenBf16LinearShapeSelector> QwenBf16LinearShapeSelector::Create(
    std::uint64_t prefill_tokens, std::uint64_t tail_prefill_tokens,
    std::uint64_t decode_tokens) {
  if (prefill_tokens == 0 || tail_prefill_tokens == 0 ||
      prefill_tokens > QwenBf16LinearShape::kMaximumTokensPerPlan ||
      tail_prefill_tokens > prefill_tokens ||
      decode_tokens != 1) {
    return Status::InvalidArgument(
        "Qwen request linear shape set is invalid");
  }
  return QwenBf16LinearShapeSelector(
      prefill_tokens, tail_prefill_tokens, decode_tokens);
}

Result<QwenBf16LinearShapeRole> QwenBf16LinearShapeSelector::select(
    std::uint64_t rows) const {
  if (rows == prefill_tokens_) return QwenBf16LinearShapeRole::kPrefill;
  if (rows == tail_prefill_tokens_) {
    return QwenBf16LinearShapeRole::kTailPrefill;
  }
  if (rows == decode_tokens_) return QwenBf16LinearShapeRole::kDecode;
  return Status::InvalidArgument(
      "Qwen linear binding does not match a frozen request shape");
}

}  // namespace pih
