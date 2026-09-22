#pragma once

#include "pih/core/buffer.h"
#include "pih/model/deepseek_fixed_state_banks.h"
#include "pih/model/deepseek_fixed_state_layout.h"

namespace pih {

class DeepSeekAttentionSequenceDeviceResources final {
 public:
  static Result<DeepSeekAttentionSequenceDeviceResources> Allocate(
      Allocator& allocator, DeepSeekFixedStateLayout fixed_layout,
      std::uintptr_t completion_event,
      DeepSeekFixedStateBankOperations& fixed_operations,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekAttentionSequenceDeviceResources(
      const DeepSeekAttentionSequenceDeviceResources&) = delete;
  DeepSeekAttentionSequenceDeviceResources& operator=(
      const DeepSeekAttentionSequenceDeviceResources&) = delete;
  DeepSeekAttentionSequenceDeviceResources(
      DeepSeekAttentionSequenceDeviceResources&&) noexcept = default;
  DeepSeekAttentionSequenceDeviceResources& operator=(
      DeepSeekAttentionSequenceDeviceResources&&) noexcept = default;

  [[nodiscard]] const DeepSeekFixedStateLayout& fixed_layout() const noexcept {
    return fixed_layout_;
  }
  [[nodiscard]] DeepSeekFixedStateBanks& fixed_banks() noexcept {
    return fixed_banks_;
  }
 private:
  DeepSeekAttentionSequenceDeviceResources(
      Buffer fixed_first, Buffer fixed_second,
      DeepSeekFixedStateLayout fixed_layout,
      DeepSeekFixedStateBanks fixed_banks) noexcept
      : fixed_first_(std::move(fixed_first)),
        fixed_second_(std::move(fixed_second)),
        fixed_layout_(std::move(fixed_layout)),
        fixed_banks_(std::move(fixed_banks)) {}

  Buffer fixed_first_;
  Buffer fixed_second_;
  DeepSeekFixedStateLayout fixed_layout_;
  DeepSeekFixedStateBanks fixed_banks_;
};

}  // namespace pih
