#include "pih/model/qwen3_kv_address_mapper.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenKvAddressMapper> QwenKvAddressMapper::Create(
    std::uint32_t slot_count, std::uint64_t backing_bytes) {
  if (slot_count == 0 || slot_count > QwenKvSlotPool::kMaximumSlots) {
    return Status::InvalidArgument("Qwen KV mapper slot count is invalid");
  }
  auto required =
      checked_mul_u64(slot_count, QwenKvSlotPool::kSlotPayloadBytes);
  if (!required.ok()) return required.status();
  if (backing_bytes < required.value() ||
      backing_bytes - required.value() >= QwenKvSlotPool::kSlotPayloadBytes) {
    return Status::InvalidArgument("Qwen KV mapper backing extent is invalid");
  }
  return QwenKvAddressMapper(slot_count, backing_bytes);
}

Result<QwenKvElementAddress> QwenKvAddressMapper::map(
    QwenKvBlockHandle handle, const QwenKvSlotState& state,
    std::uint32_t owner_sequence_index, std::uint32_t layer,
    QwenKvPlane plane, std::uint32_t token_in_slot, std::uint32_t kv_head,
    std::uint32_t head_column) const {
  if (handle.slot >= slot_count_ || handle.generation == 0 ||
      state.generation != handle.generation ||
      state.owner_sequence_index != owner_sequence_index ||
      state.state != QwenKvSlotLifecycle::kOwned ||
      state.reserved_zero_u8 != 0 || state.reserved_zero_u32 != 0) {
    return Status::FailedPrecondition(
        "Qwen KV mapper rejected stale or unowned handle");
  }
  if (layer >= QwenKvSlotPool::kLayerCount ||
      (plane != QwenKvPlane::kKey && plane != QwenKvPlane::kValue) ||
      token_in_slot >= state.valid_tokens ||
      token_in_slot >= QwenKvSlotPool::kTokensPerSlot ||
      kv_head >= kKvHeads || head_column >= kHeadDimension) {
    return Status::InvalidArgument("Qwen KV logical coordinate is invalid");
  }
  std::uint64_t offset =
      static_cast<std::uint64_t>(handle.slot) *
      QwenKvSlotPool::kSlotPayloadBytes;
  offset += static_cast<std::uint64_t>(layer) * kBytesPerLayer;
  if (plane == QwenKvPlane::kValue) offset += kBytesPerPlane;
  offset += static_cast<std::uint64_t>(token_in_slot) * kBytesPerToken;
  offset += static_cast<std::uint64_t>(kv_head) * kHeadDimension *
            kElementBytes;
  offset += static_cast<std::uint64_t>(head_column) * kElementBytes;
  auto end = checked_add_u64(offset, kElementBytes);
  if (!end.ok()) return end.status();
  if (end.value() > backing_bytes_) {
    return Status::Internal("Qwen KV mapped address exceeds backing");
  }
  return QwenKvElementAddress{offset, kElementBytes, handle.slot,
                              handle.generation};
}

}  // namespace pih
