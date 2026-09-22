#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

class DeepSeekCompressorPoolingOracle final {
 public:
  static Result<DeepSeekCompressorPoolingOracle> Create(
      std::uint32_t ratio, std::uint32_t head_dim);

  // Inputs are the FP32 wkv/wgate projections and the matching APE row for
  // one source token. A non-empty result is produced only at ratio boundaries,
  // before RMSNorm, RoPE, optional Hadamard rotation, and activation quant.
  Result<std::vector<float>> step(
      std::uint32_t absolute_position,
      std::span<const float> kv_projection,
      std::span<const float> gate_projection,
      std::span<const float> ape_row);

  [[nodiscard]] std::uint32_t remainder_tokens() const noexcept {
    return remainder_tokens_;
  }
  [[nodiscard]] std::uint64_t completed_slots() const noexcept {
    return completed_slots_;
  }

 private:
  std::uint32_t ratio_ = 0;
  std::uint32_t head_dim_ = 0;
  std::uint32_t projection_dim_ = 0;
  std::uint32_t remainder_tokens_ = 0;
  std::uint64_t completed_slots_ = 0;
  std::vector<float> kv_state_;
  std::vector<float> score_state_;
};

}  // namespace pih
