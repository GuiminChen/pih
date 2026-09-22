#include "pih/model/qwen3_bf16_mlp_arena.h"

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {

Result<QwenBf16MlpArenaLayout> QwenBf16MlpArenaLayout::Create(
    std::uint64_t tokens) {
  auto gate = QwenBf16LinearShape::Create(QwenBf16LinearKind::kGate, tokens);
  if (!gate.ok()) return gate.status();
  auto elements = checked_mul_u64(tokens, kIntermediateFeatures);
  if (!elements.ok()) return elements.status();
  auto bytes = checked_mul_u64(elements.value(), kElementBytes);
  if (!bytes.ok()) return bytes.status();
  auto up_offset = checked_align_up_u64(bytes.value(), kAlignment);
  if (!up_offset.ok()) return up_offset.status();
  auto end = checked_add_u64(up_offset.value(), bytes.value());
  if (!end.ok()) return end.status();
  auto arena = checked_align_up_u64(end.value(), kAlignment);
  if (!arena.ok()) return arena.status();
  return QwenBf16MlpArenaLayout(tokens, bytes.value(), up_offset.value(),
                                arena.value());
}

}  // namespace pih
