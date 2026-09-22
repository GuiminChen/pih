#include "pih/model/qwen3_paged_kv_oracle.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <unordered_set>
#include <vector>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_kv_address_mapper.h"
#include "pih/model/qwen3_kv_block_table.h"

namespace pih {

Status qwen_materialize_paged_kv_oracle(
    std::span<const BFloat16> physical_backing,
    std::span<const QwenKvSlotState> slot_states,
    std::span<const QwenKvBlockHandle> handles,
    std::uint32_t owner_sequence_index, std::uint32_t layer,
    std::uint32_t token_count, std::span<BFloat16> key_output,
    std::span<BFloat16> value_output) {
  if (slot_states.empty() ||
      slot_states.size() > QwenKvSlotPool::kMaximumSlots || token_count == 0 ||
      token_count > QwenKvBlockTable::kMaximumReservedTokens ||
      layer >= QwenKvSlotPool::kLayerCount) {
    return Status::InvalidArgument("Qwen paged KV oracle range is invalid");
  }
  const std::uint32_t required_handles =
      (token_count + QwenKvSlotPool::kTokensPerSlot - 1U) /
      QwenKvSlotPool::kTokensPerSlot;
  if (handles.size() != required_handles || handles.size() > slot_states.size()) {
    return Status::InvalidArgument("Qwen paged KV handle extent is invalid");
  }
  auto elements = checked_mul_u64(token_count,
                                  QwenKvAddressMapper::kKvHeads);
  if (!elements.ok()) return elements.status();
  elements = checked_mul_u64(elements.value(),
                             QwenKvAddressMapper::kHeadDimension);
  if (!elements.ok()) return elements.status();
  if (elements.value() > std::numeric_limits<std::size_t>::max() ||
      key_output.size() != elements.value() ||
      value_output.size() != elements.value()) {
    return Status::InvalidArgument("Qwen paged KV output extent is invalid");
  }
  auto backing_bytes = checked_mul_u64(physical_backing.size(),
                                       sizeof(BFloat16));
  if (!backing_bytes.ok()) return backing_bytes.status();
  auto mapper = QwenKvAddressMapper::Create(
      static_cast<std::uint32_t>(slot_states.size()), backing_bytes.value());
  if (!mapper.ok()) return mapper.status();
  std::unordered_set<std::uint32_t> unique_slots;
  unique_slots.reserve(handles.size());
  for (const auto handle : handles) {
    if (!unique_slots.insert(handle.slot).second) {
      return Status::InvalidArgument("Qwen paged KV handles repeat a slot");
    }
  }

  std::vector<BFloat16> keys(key_output.size());
  std::vector<BFloat16> values(value_output.size());
  for (std::uint32_t token = 0; token < token_count; ++token) {
    const auto handle = handles[token / QwenKvSlotPool::kTokensPerSlot];
    if (handle.slot >= slot_states.size()) {
      return Status::FailedPrecondition("Qwen paged KV handle slot is stale");
    }
    const auto& state = slot_states[handle.slot];
    const std::uint32_t token_in_slot =
        token % QwenKvSlotPool::kTokensPerSlot;
    for (std::uint32_t head = 0; head < QwenKvAddressMapper::kKvHeads;
         ++head) {
      for (std::uint32_t column = 0;
           column < QwenKvAddressMapper::kHeadDimension; ++column) {
        auto key_address = mapper->map(
            handle, state, owner_sequence_index, layer, QwenKvPlane::kKey,
            token_in_slot, head, column);
        auto value_address = mapper->map(
            handle, state, owner_sequence_index, layer, QwenKvPlane::kValue,
            token_in_slot, head, column);
        if (!key_address.ok()) return key_address.status();
        if (!value_address.ok()) return value_address.status();
        const std::size_t output_index =
            (static_cast<std::size_t>(token) *
                 QwenKvAddressMapper::kKvHeads +
             head) *
                QwenKvAddressMapper::kHeadDimension +
            column;
        keys[output_index] =
            physical_backing[key_address->byte_offset / sizeof(BFloat16)];
        values[output_index] =
            physical_backing[value_address->byte_offset / sizeof(BFloat16)];
      }
    }
  }
  std::copy(keys.begin(), keys.end(), key_output.begin());
  std::copy(values.begin(), values.end(), value_output.begin());
  return Status::Ok();
}

}  // namespace pih
