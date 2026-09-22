#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

struct DeepSeekExpertArenaSpan final {
  std::uintptr_t address = 0;
  std::uint64_t bytes = 0;
};

struct DeepSeekExpertComputeArena final {
  DeepSeekExpertArenaSpan route_input_bf16;
  DeepSeekExpertArenaSpan expert_output_bf16;  // aliases route_input after W1/W3
  DeepSeekExpertArenaSpan activation_e4m3;
  DeepSeekExpertArenaSpan activation_scale_bits;
  DeepSeekExpertArenaSpan gate_or_middle_bf16;  // in-place SwiGLU
  DeepSeekExpertArenaSpan up_bf16;
  DeepSeekExpertArenaSpan route_weights_f32;
  DeepSeekExpertArenaSpan token_indices_u32;
  DeepSeekExpertArenaSpan error_flag_u32;
};

class DeepSeekExpertComputeArenaLayout final {
 public:
  static constexpr std::uint32_t kMaximumTokens = 4096;
  static constexpr std::uint64_t kAlignment = 256;

  static Result<DeepSeekExpertComputeArenaLayout> Create(
      std::uint32_t maximum_tokens);
  Result<DeepSeekExpertComputeArena> bind(std::uintptr_t base,
                                          std::uint64_t backing_bytes) const;
  [[nodiscard]] std::uint64_t required_bytes() const noexcept {
    return required_bytes_;
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }

 private:
  struct Region final {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };
  std::uint32_t maximum_tokens_ = 0;
  std::uint64_t required_bytes_ = 0;
  Region route_input_output_;
  Region activation_;
  Region activation_scales_;
  Region gate_middle_;
  Region up_;
  Region route_weights_;
  Region token_indices_;
  Region error_flag_;
};

}  // namespace pih
