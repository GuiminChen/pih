#pragma once

#include "pih/core/buffer.h"
#include "pih/model/deepseek_expert_compute_arena.h"
#include "pih/model/deepseek_learned_router_coordinator.h"

namespace pih {

struct DeepSeekLearnedRouterDeviceView final {
  std::uintptr_t scores_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
};

class DeepSeekLearnedRouterDeviceResources final {
 public:
  static Result<DeepSeekLearnedRouterDeviceResources> Allocate(
      std::uint32_t maximum_tokens, DeepSeekExpertComputeArena expert_arena,
      Allocator& allocator, std::uint64_t context_identity,
      std::int32_t device_ordinal);

  [[nodiscard]] DeepSeekLearnedRouterDeviceView view() const noexcept {
    return view_;
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }
  [[nodiscard]] std::uint64_t scores_bytes() const noexcept {
    return scores_.size_bytes();
  }

 private:
  DeepSeekLearnedRouterDeviceResources(
      Buffer scores, DeepSeekLearnedRouterDeviceView view,
      std::uint32_t maximum_tokens) noexcept
      : scores_(std::move(scores)), view_(view),
        maximum_tokens_(maximum_tokens) {}

  Buffer scores_;
  DeepSeekLearnedRouterDeviceView view_;
  std::uint32_t maximum_tokens_ = 0;
};

}  // namespace pih
