#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {

Result<std::uint64_t> qwen_bf16_linear_execution_rows(
    QwenBf16LinearKind kind, std::uint64_t packed_tokens,
    std::uint64_t active_logit_rows) {
  if (packed_tokens == 0 || packed_tokens > 4096) {
    return Status::InvalidArgument(
        "Qwen BF16 packed token rows are outside the plan bound");
  }
  if (active_logit_rows == 0 || active_logit_rows > packed_tokens) {
    return Status::InvalidArgument(
        "Qwen BF16 active logit rows must be within packed token rows");
  }
  switch (kind) {
    case QwenBf16LinearKind::kQuery:
    case QwenBf16LinearKind::kKey:
    case QwenBf16LinearKind::kValue:
    case QwenBf16LinearKind::kAttentionOutput:
    case QwenBf16LinearKind::kGate:
    case QwenBf16LinearKind::kUp:
    case QwenBf16LinearKind::kDown:
      return packed_tokens;
    case QwenBf16LinearKind::kLmHead:
      return active_logit_rows;
  }
  return Status::InvalidArgument("unknown Qwen BF16 linear kind");
}

Result<QwenBf16LinearShape> QwenBf16LinearShape::Create(
    QwenBf16LinearKind kind, std::uint64_t tokens) {
  if (tokens == 0 || tokens > kMaximumTokensPerPlan) {
    return Status::InvalidArgument(
        "Qwen BF16 linear token count is outside the frozen plan bound");
  }
  switch (kind) {
    case QwenBf16LinearKind::kQuery:
      return QwenBf16LinearShape(kind, "qwen.linear.query.bf16.v1", tokens,
                                 1024, 2048);
    case QwenBf16LinearKind::kKey:
      return QwenBf16LinearShape(kind, "qwen.linear.key.bf16.v1", tokens,
                                 1024, 1024);
    case QwenBf16LinearKind::kValue:
      return QwenBf16LinearShape(kind, "qwen.linear.value.bf16.v1", tokens,
                                 1024, 1024);
    case QwenBf16LinearKind::kAttentionOutput:
      return QwenBf16LinearShape(
          kind, "qwen.linear.attention_output.bf16.v1", tokens, 2048, 1024);
    case QwenBf16LinearKind::kGate:
      return QwenBf16LinearShape(kind, "qwen.linear.gate.bf16.v1", tokens,
                                 1024, 3072);
    case QwenBf16LinearKind::kUp:
      return QwenBf16LinearShape(kind, "qwen.linear.up.bf16.v1", tokens,
                                 1024, 3072);
    case QwenBf16LinearKind::kDown:
      return QwenBf16LinearShape(kind, "qwen.linear.down.bf16.v1", tokens,
                                 3072, 1024);
    case QwenBf16LinearKind::kLmHead:
      return QwenBf16LinearShape(kind, "qwen.linear.lm_head.bf16.v1", tokens,
                                 1024, 151936);
  }
  return Status::InvalidArgument("unknown Qwen BF16 linear kind");
}

}  // namespace pih
