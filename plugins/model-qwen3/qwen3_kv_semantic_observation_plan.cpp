#include "pih/model/qwen3_kv_semantic_observation_plan.h"

#include <algorithm>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_semantic_outcome_recorder.h"

namespace pih {

Result<QwenKvSemanticObservationPlan>
QwenKvSemanticObservationPlan::Create(
    const QwenKvBlockTable& table,
    std::span<const QwenKvSlotState> slot_states,
    std::uint64_t backing_bytes) {
  const auto& descriptor = table.descriptor();
  const auto handles = table.visible_handles();
  if (descriptor.active != 1 || descriptor.committed_tokens == 0 ||
      handles.empty() || handles.size() > descriptor.handle_count ||
      slot_states.empty() ||
      slot_states.size() > QwenKvSlotPool::kMaximumSlots) {
    return Status::FailedPrecondition(
        "Qwen KV semantic observation has invalid committed identity");
  }
  auto mapper = QwenKvAddressMapper::Create(
      static_cast<std::uint32_t>(slot_states.size()), backing_bytes);
  if (!mapper.ok()) return mapper.status();
  auto token_plane_bytes = checked_mul_u64(
      descriptor.committed_tokens, QwenKvAddressMapper::kBytesPerToken);
  if (!token_plane_bytes.ok()) return token_plane_bytes.status();
  auto payload_bytes = checked_mul_u64(
      *token_plane_bytes, 2ULL * QwenKvSlotPool::kLayerCount);
  if (!payload_bytes.ok()) return payload_bytes.status();
  if (*payload_bytes > QwenSemanticOutcomeRecorder::kMaximumKvStateBytes) {
    return Status::ResourceExhausted(
        "Qwen KV semantic observation exceeds evidence payload bound");
  }

  std::vector<QwenKvSemanticCopySlice> slices;
  slices.reserve(handles.size() * 2 * QwenKvSlotPool::kLayerCount);
  for (std::uint32_t layer = 0; layer < QwenKvSlotPool::kLayerCount;
       ++layer) {
    for (const auto plane : {QwenKvPlane::kKey, QwenKvPlane::kValue}) {
      const std::uint64_t plane_index =
          static_cast<std::uint64_t>(layer) * 2 +
          static_cast<std::uint8_t>(plane);
      for (std::size_t logical_block = 0; logical_block < handles.size();
           ++logical_block) {
        const auto handle = handles[logical_block];
        if (handle.slot >= slot_states.size()) {
          return Status::FailedPrecondition(
              "Qwen KV semantic observation handle exceeds state table");
        }
        const auto& state = slot_states[handle.slot];
        const std::uint32_t first_token = static_cast<std::uint32_t>(
            logical_block * QwenKvSlotPool::kTokensPerSlot);
        const std::uint32_t remaining =
            descriptor.committed_tokens - first_token;
        const std::uint16_t expected_valid = static_cast<std::uint16_t>(
            std::min<std::uint32_t>(remaining,
                                    QwenKvSlotPool::kTokensPerSlot));
        if (state.generation != handle.generation ||
            state.owner_sequence_index != descriptor.owner_sequence_index ||
            state.state != QwenKvSlotLifecycle::kOwned ||
            state.valid_tokens != expected_valid ||
            state.reserved_zero_u8 != 0 || state.reserved_zero_u32 != 0) {
          return Status::FailedPrecondition(
              "Qwen KV semantic observation rejected stale slot state");
        }
        auto mapped = mapper->map(
            handle, state, descriptor.owner_sequence_index, layer, plane, 0,
            0, 0);
        if (!mapped.ok()) return mapped.status();
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(expected_valid) *
            QwenKvAddressMapper::kBytesPerToken;
        const std::uint64_t destination_offset =
            plane_index * *token_plane_bytes +
            static_cast<std::uint64_t>(first_token) *
                QwenKvAddressMapper::kBytesPerToken;
        slices.push_back({mapped->byte_offset, destination_offset, bytes,
                          static_cast<std::uint32_t>(logical_block), layer,
                          plane});
      }
    }
  }
  return QwenKvSemanticObservationPlan(*payload_bytes, std::move(slices));
}

}  // namespace pih
