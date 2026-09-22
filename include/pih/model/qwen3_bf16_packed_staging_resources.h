#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "pih/core/tensor_view.h"
#include "pih/model/qwen3_bf16_packed_step_staging_layout.h"
#include "pih/model/qwen3_bf16_step_resource_factory.h"

namespace pih {

enum class QwenBf16PackedStagingSlot : std::uint8_t {
  kTokenIds = 0,
  kPositions,
  kRequestIndex,
  kQueryStartOffsets,
  kSampleRowIndex,
  kSampleCount,
  kKvAppendHandles,
  kKvTokenOffsets,
  kKvVisibleHandleOffsets,
  kKvVisibleHandles,
  kKeyTokenCounts,
  kOwnerSequenceIndices,
  kSamplingDescriptors,
  kSampleSequenceIndices,
  kCount,
};

class QwenBf16PackedStagingResources final {
 public:
  static constexpr std::size_t kSlotCount =
      static_cast<std::size_t>(QwenBf16PackedStagingSlot::kCount);

  static Result<QwenBf16PackedStagingResources> Create(
      std::uint64_t request_generation, std::int32_t owning_rank,
      const QwenBf16PackedStepStagingLayout& layout,
      QwenBf16DeviceArenaOwner owner);

  Result<TensorView> view(QwenBf16PackedStagingSlot slot) const;

  [[nodiscard]] std::uint64_t request_generation() const noexcept {
    return request_generation_;
  }
  [[nodiscard]] std::int32_t owning_rank() const noexcept {
    return owning_rank_;
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return kSlotCount;
  }

 private:
  std::uint64_t request_generation_ = 0;
  std::int32_t owning_rank_ = -1;
  std::array<std::optional<TensorView>, kSlotCount> resources_{};
};

}  // namespace pih
