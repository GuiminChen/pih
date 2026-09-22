#pragma once

#include <memory>

#include "pih/core/buffer.h"

namespace pih {

struct DeepSeekMhcDeviceView final {
  std::uintptr_t residual_a_bf16 = 0;
  std::uintptr_t residual_b_bf16 = 0;
  std::uintptr_t layer_input_bf16 = 0;
  std::uintptr_t ffn_branch_output_bf16 = 0;
  std::uintptr_t post_mix_f32 = 0;
  std::uintptr_t residual_mix_f32 = 0;
};

class DeepSeekMhcDeviceResources final {
 public:
  static Result<DeepSeekMhcDeviceResources> Allocate(
      Allocator& allocator, std::uint32_t maximum_tokens,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekMhcDeviceResources(DeepSeekMhcDeviceResources&&) noexcept = default;
  DeepSeekMhcDeviceResources& operator=(DeepSeekMhcDeviceResources&&) noexcept =
      default;
  DeepSeekMhcDeviceResources(const DeepSeekMhcDeviceResources&) = delete;
  DeepSeekMhcDeviceResources& operator=(const DeepSeekMhcDeviceResources&) =
      delete;

  [[nodiscard]] DeepSeekMhcDeviceView view() const noexcept { return view_; }
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
  DeepSeekMhcDeviceResources(std::unique_ptr<Buffer> backing,
                             DeepSeekMhcDeviceView view,
                             std::uint32_t maximum_tokens,
                             std::uint64_t context_identity,
                             std::int32_t device_ordinal) noexcept
      : backing_(std::move(backing)), view_(view),
        maximum_tokens_(maximum_tokens), context_identity_(context_identity),
        device_ordinal_(device_ordinal) {}

  std::unique_ptr<Buffer> backing_;
  DeepSeekMhcDeviceView view_;
  std::uint32_t maximum_tokens_ = 0;
  std::uint64_t context_identity_ = 0;
  std::int32_t device_ordinal_ = -1;
};

}  // namespace pih
