#pragma once

#include <memory>

#include "pih/core/buffer.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

struct DeepSeekDsparkDeviceView final {
  std::uintptr_t draft_token_ids_u32 = 0;
  std::uintptr_t draft_input_hc_bf16 = 0;
  std::uintptr_t main_quant_fp8 = 0;
  std::uintptr_t main_quant_scale_ue8m0 = 0;
  std::uintptr_t main_projected_bf16 = 0;
  std::uintptr_t main_normalized_bf16 = 0;
  std::uintptr_t head_hidden_bf16 = 0;
  std::uintptr_t normalized_bf16 = 0;
  std::uintptr_t raw_logits_f32 = 0;
  std::uintptr_t biased_logits_f32 = 0;
  std::uintptr_t markov_embeddings_bf16 = 0;
  std::uintptr_t confidence_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  // Concatenated means of target layers 40, 41 and 42:
  // [maximum_tokens, 3, 4096] BF16.
  std::uintptr_t target_hidden_bf16 = 0;
  // Shared sequential scratch for one MTP stage's [token, 512] KV projection.
  std::uintptr_t prefill_kv_bf16 = 0;
  // Three stages execute serially.  The embed output is buffer zero and these
  // two fixed block buffers carry stage 0->1 and stage 1->2 residuals.
  std::uintptr_t stage_residual_a_bf16 = 0;
  std::uintptr_t stage_residual_b_bf16 = 0;
  std::uintptr_t draft_positions_u32 = 0;
};

class DeepSeekDsparkDeviceResources final {
 public:
  static Result<DeepSeekDsparkDeviceResources> Allocate(
      DeepSeekStagePlan stage, std::uint32_t maximum_tokens,
      Allocator& allocator,
      std::uint64_t context_identity, std::int32_t device_ordinal);
  [[nodiscard]] DeepSeekDsparkDeviceView view() const noexcept { return view_; }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_ == nullptr ? 0 : backing_->size_bytes();
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }

 private:
  DeepSeekDsparkDeviceResources(std::unique_ptr<Buffer> backing,
                               DeepSeekDsparkDeviceView view,
                               std::uint32_t maximum_tokens) noexcept
      : backing_(std::move(backing)), view_(view),
        maximum_tokens_(maximum_tokens) {}
  std::unique_ptr<Buffer> backing_;
  DeepSeekDsparkDeviceView view_;
  std::uint32_t maximum_tokens_ = 0;
};

}  // namespace pih
