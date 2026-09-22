#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_kv_address_mapper.h"
#include "pih/model/qwen3_kv_block_table.h"

namespace pih {

struct QwenKvSemanticCopySlice final {
  std::uint64_t source_offset;
  std::uint64_t destination_offset;
  std::uint64_t bytes;
  std::uint32_t logical_block;
  std::uint32_t layer;
  QwenKvPlane plane;
};

class QwenKvSemanticObservationPlan final {
 public:
  static Result<QwenKvSemanticObservationPlan> Create(
      const QwenKvBlockTable& table,
      std::span<const QwenKvSlotState> slot_states,
      std::uint64_t backing_bytes);

  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::span<const QwenKvSemanticCopySlice> slices()
      const noexcept {
    return slices_;
  }

 private:
  QwenKvSemanticObservationPlan(
      std::uint64_t payload_bytes,
      std::vector<QwenKvSemanticCopySlice> slices)
      : payload_bytes_(payload_bytes), slices_(std::move(slices)) {}

  std::uint64_t payload_bytes_;
  std::vector<QwenKvSemanticCopySlice> slices_;
};

}  // namespace pih
