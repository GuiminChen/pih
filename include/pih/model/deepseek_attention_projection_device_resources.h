#pragma once

#include <memory>

#include "pih/core/buffer.h"

namespace pih {

struct DeepSeekAttentionProjectionDeviceView final {
  std::uintptr_t input_e4m3 = 0;
  std::uintptr_t input_scale_ue8m0 = 0;
  std::uintptr_t q_a_bf16 = 0;
  std::uintptr_t q_norm_bf16 = 0;
  std::uintptr_t q_e4m3 = 0;
  std::uintptr_t q_scale_ue8m0 = 0;
  std::uintptr_t query_bf16 = 0;
  std::uintptr_t kv_bf16 = 0;
  std::uintptr_t attention_output_bf16 = 0;
  std::uintptr_t wo_a_activation_e4m3 = 0;
  std::uintptr_t wo_a_activation_scale_ue8m0 = 0;
  std::uintptr_t wo_a_output_bf16 = 0;
  std::uintptr_t wo_b_activation_e4m3 = 0;
  std::uintptr_t wo_b_activation_scale_ue8m0 = 0;
  std::uintptr_t branch_output_bf16 = 0;
  std::uintptr_t token_ids_u32 = 0;
  std::uintptr_t positions_u32 = 0;
  std::uintptr_t error_flag_u32 = 0;
};

class DeepSeekAttentionProjectionDeviceResources final {
 public:
  static Result<DeepSeekAttentionProjectionDeviceResources> Allocate(
      Allocator& allocator, std::uint32_t maximum_tokens,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekAttentionProjectionDeviceResources(
      DeepSeekAttentionProjectionDeviceResources&&) noexcept = default;
  DeepSeekAttentionProjectionDeviceResources& operator=(
      DeepSeekAttentionProjectionDeviceResources&&) noexcept = default;
  DeepSeekAttentionProjectionDeviceResources(
      const DeepSeekAttentionProjectionDeviceResources&) = delete;
  DeepSeekAttentionProjectionDeviceResources& operator=(
      const DeepSeekAttentionProjectionDeviceResources&) = delete;

  [[nodiscard]] DeepSeekAttentionProjectionDeviceView view() const noexcept {
    return view_;
  }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_ == nullptr ? 0 : backing_->size_bytes();
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }
  [[nodiscard]] std::uint64_t context_identity() const noexcept {
    return context_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }

 private:
  DeepSeekAttentionProjectionDeviceResources(
      std::unique_ptr<Buffer> backing,
      DeepSeekAttentionProjectionDeviceView view,
      std::uint32_t maximum_tokens, std::uint64_t context_identity,
      std::int32_t device_ordinal) noexcept
      : backing_(std::move(backing)), view_(view),
        maximum_tokens_(maximum_tokens), context_identity_(context_identity),
        device_ordinal_(device_ordinal) {}

  std::unique_ptr<Buffer> backing_;
  DeepSeekAttentionProjectionDeviceView view_;
  std::uint32_t maximum_tokens_ = 0;
  std::uint64_t context_identity_ = 0;
  std::int32_t device_ordinal_ = -1;
};

}  // namespace pih
