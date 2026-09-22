#include "pih/model/qwen3_bf16_execution_arena.h"

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {
namespace {

Result<QwenBf16ArenaSpan> append_span(std::uint64_t* next,
                                     std::uint64_t tokens,
                                     std::uint64_t features) {
  auto elements = checked_mul_u64(tokens, features);
  if (!elements.ok()) return elements.status();
  auto bytes = checked_mul_u64(*elements, 2);
  if (!bytes.ok()) return bytes.status();
  auto offset = checked_align_up_u64(*next, 256);
  if (!offset.ok()) return offset.status();
  auto end = checked_add_u64(*offset, *bytes);
  if (!end.ok()) return end.status();
  *next = *end;
  return QwenBf16ArenaSpan{*offset, *bytes};
}

}  // namespace

Result<QwenBf16ExecutionArenaLayout> QwenBf16ExecutionArenaLayout::Create(
    std::uint64_t tokens, std::uint64_t active_logit_rows) {
  if (tokens == 0 ||
      tokens > QwenBf16LinearShape::kMaximumTokensPerPlan) {
    return Status::InvalidArgument(
        "Qwen BF16 execution arena token count is outside the plan bound");
  }
  if (active_logit_rows == 0 || active_logit_rows > tokens) {
    return Status::InvalidArgument(
        "Qwen BF16 active logit rows must be within packed token rows");
  }

  std::uint64_t next = 0;
  auto hidden = append_span(&next, tokens, 1024);
  if (!hidden.ok()) return hidden.status();
  auto normalized = append_span(&next, tokens, 1024);
  if (!normalized.ok()) return normalized.status();
  auto query = append_span(&next, tokens, 2048);
  if (!query.ok()) return query.status();
  auto key = append_span(&next, tokens, 1024);
  if (!key.ok()) return key.status();
  auto value = append_span(&next, tokens, 1024);
  if (!value.ok()) return value.status();
  auto attention = append_span(&next, tokens, 2048);
  if (!attention.ok()) return attention.status();
  auto activation_bytes = checked_align_up_u64(next, kAlignment);
  if (!activation_bytes.ok()) return activation_bytes.status();

  auto mlp = QwenBf16MlpArenaLayout::Create(tokens);
  if (!mlp.ok()) return mlp.status();
  auto rope_bytes = checked_mul_u64(tokens, 512);
  if (!rope_bytes.ok()) return rope_bytes.status();
  auto logits = checked_mul_u64(active_logit_rows, kVocabularySize);
  if (!logits.ok()) return logits.status();
  logits = checked_mul_u64(*logits, kFp32Bytes);
  if (!logits.ok()) return logits.status();

  return QwenBf16ExecutionArenaLayout(
      tokens, active_logit_rows, *hidden, *normalized, *query, *key, *value,
      *attention, *activation_bytes, *mlp, *rope_bytes, *logits);
}

}  // namespace pih
