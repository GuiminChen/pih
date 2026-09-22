#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "pih/core/result.h"

namespace pih {
struct DeepSeekDsparkGreedyVerifyResult final {
  std::array<std::uint32_t, 5> retained_tokens{};
  std::uint32_t accepted_draft_count = 0;
  std::uint32_t retained_record_count = 0;
  bool mismatch = false;
};
Result<DeepSeekDsparkGreedyVerifyResult> deepseek_dspark_greedy_verify(
    std::span<const std::uint32_t> draft_tokens,
    std::span<const std::uint32_t> target_argmax_tokens,
    std::uint32_t vocab_size = 129280);
}  // namespace pih
