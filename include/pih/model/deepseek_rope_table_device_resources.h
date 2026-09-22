#pragma once

#include "pih/backend/cuda/deepseek_rope_table.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

struct DeepSeekRopeTableDeviceView final {
  std::uintptr_t base_f32 = 0;
  std::uintptr_t yarn_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
};

class DeepSeekRopeTableDeviceResources final {
 public:
  static Result<DeepSeekRopeTableDeviceResources> AllocateDeferred(
      Allocator& allocator, DeepSeekStagePlan stage,
      std::uint32_t maximum_positions, std::uintptr_t stream,
      DeepSeekRopeTableOperations& operations,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  Status ensure_initialized();
  Status retire_pending() noexcept;

  DeepSeekRopeTableDeviceResources(
      const DeepSeekRopeTableDeviceResources&) = delete;
  DeepSeekRopeTableDeviceResources& operator=(
      const DeepSeekRopeTableDeviceResources&) = delete;
  DeepSeekRopeTableDeviceResources(
      DeepSeekRopeTableDeviceResources&&) noexcept = default;
  DeepSeekRopeTableDeviceResources& operator=(
      DeepSeekRopeTableDeviceResources&&) noexcept = default;

  [[nodiscard]] DeepSeekRopeTableDeviceView view() const noexcept {
    return view_;
  }
  [[nodiscard]] std::uint32_t maximum_positions() const noexcept {
    return maximum_positions_;
  }
  [[nodiscard]] std::uint64_t context_identity() const noexcept {
    return context_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_.size_bytes();
  }

 private:
  DeepSeekRopeTableDeviceResources(
      Buffer backing, DeepSeekRopeTableDeviceView view,
      std::uint32_t maximum_positions, std::uint64_t context_identity,
      std::int32_t device_ordinal, DeepSeekRopeTableOperations& operations,
      std::uintptr_t stream) noexcept
      : backing_(std::move(backing)), view_(view),
        maximum_positions_(maximum_positions),
        context_identity_(context_identity), device_ordinal_(device_ordinal),
        operations_(&operations), stream_(stream) {}

  Buffer backing_;
  DeepSeekRopeTableDeviceView view_{};
  std::uint32_t maximum_positions_ = 0;
  std::uint64_t context_identity_ = 0;
  std::int32_t device_ordinal_ = -1;
  DeepSeekRopeTableOperations* operations_ = nullptr;
  std::uintptr_t stream_ = 0;
  bool pending_ = false;
  bool initialized_ = false;
};

}  // namespace pih
