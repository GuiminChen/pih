#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "pih/core/result.h"

namespace pih {

struct Qwen3Config final {
  static constexpr std::size_t kMaxConfigBytes = 1024 * 1024;

  static Result<Qwen3Config> Parse(std::string_view json);

  std::uint64_t hidden_size;
  std::uint64_t intermediate_size;
  std::uint64_t layers;
  std::uint64_t attention_heads;
  std::uint64_t kv_heads;
  std::uint64_t head_dim;
  std::uint64_t vocabulary_size;
  std::uint64_t maximum_positions;
  double rope_theta;
  double rms_norm_epsilon;
  std::int64_t bos_token_id;
  std::int64_t eos_token_id;

  [[nodiscard]] std::uint64_t q_projection_size() const noexcept {
    return attention_heads * head_dim;
  }
  [[nodiscard]] std::uint64_t kv_projection_size() const noexcept {
    return kv_heads * head_dim;
  }
};

}  // namespace pih
