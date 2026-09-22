#pragma once

#include <memory>

#include "pih/core/buffer.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

struct DeepSeekEndpointDeviceView final {
  std::uintptr_t embedding_output_hc_bf16 = 0;
  std::uintptr_t head_output_bf16 = 0;
  std::uintptr_t normalized_bf16 = 0;
  std::uintptr_t logits_f32 = 0;
  std::uintptr_t sampled_token_u32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t selected_logprob_f32 = 0;
  std::uintptr_t rng_word_u32 = 0;
  std::uintptr_t sampling_workspace_values_f32 = 0;
  std::uintptr_t sampling_workspace_ids_u32 = 0;
  std::uintptr_t top_logprobs_ids_u32 = 0;
  std::uintptr_t top_logprobs_f32 = 0;
};

class DeepSeekEndpointDeviceResources final {
 public:
  static Result<DeepSeekEndpointDeviceResources> Allocate(
      DeepSeekStagePlan stage, std::uint32_t maximum_tokens,
      Allocator& allocator, std::uint64_t context_identity,
      std::int32_t device_ordinal);

  [[nodiscard]] DeepSeekEndpointDeviceView view() const noexcept {
    return view_;
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_ == nullptr ? 0 : backing_->size_bytes();
  }

 private:
  DeepSeekEndpointDeviceResources(
      std::unique_ptr<Buffer> backing, DeepSeekEndpointDeviceView view,
      std::uint32_t maximum_tokens) noexcept
      : backing_(std::move(backing)), view_(view),
        maximum_tokens_(maximum_tokens) {}

  std::unique_ptr<Buffer> backing_;
  DeepSeekEndpointDeviceView view_;
  std::uint32_t maximum_tokens_ = 0;
};

}  // namespace pih
