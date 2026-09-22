#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_kv_slot_pool.h"

namespace pih {

enum class QwenKvPlane : std::uint8_t { kKey = 0, kValue = 1 };

struct QwenKvElementAddress final {
  std::uint64_t byte_offset;
  std::uint64_t byte_span;
  std::uint32_t slot;
  std::uint32_t generation;
};

class QwenKvAddressMapper final {
 public:
  static constexpr std::uint32_t kKvHeads = 8;
  static constexpr std::uint32_t kHeadDimension = 128;
  static constexpr std::uint64_t kElementBytes = 2;
  static constexpr std::uint64_t kBytesPerToken =
      kKvHeads * kHeadDimension * kElementBytes;
  static constexpr std::uint64_t kBytesPerPlane =
      QwenKvSlotPool::kTokensPerSlot * kBytesPerToken;
  static constexpr std::uint64_t kBytesPerLayer = 2 * kBytesPerPlane;

  static Result<QwenKvAddressMapper> Create(std::uint32_t slot_count,
                                            std::uint64_t backing_bytes);

  Result<QwenKvElementAddress> map(
      QwenKvBlockHandle handle, const QwenKvSlotState& state,
      std::uint32_t owner_sequence_index, std::uint32_t layer,
      QwenKvPlane plane, std::uint32_t token_in_slot, std::uint32_t kv_head,
      std::uint32_t head_column) const;

  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return slot_count_;
  }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_bytes_;
  }

 private:
  QwenKvAddressMapper(std::uint32_t slot_count, std::uint64_t backing_bytes)
      : slot_count_(slot_count), backing_bytes_(backing_bytes) {}

  std::uint32_t slot_count_;
  std::uint64_t backing_bytes_;
};

static_assert(QwenKvAddressMapper::kBytesPerPlane == 32 * 1024);
static_assert(QwenKvAddressMapper::kBytesPerLayer == 64 * 1024);
static_assert(QwenKvAddressMapper::kBytesPerLayer *
                  QwenKvSlotPool::kLayerCount ==
              QwenKvSlotPool::kSlotPayloadBytes);

}  // namespace pih
