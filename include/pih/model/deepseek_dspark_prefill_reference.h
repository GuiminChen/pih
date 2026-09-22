#pragma once

#include <array>
#include <span>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"
#include "pih/model/deepseek_dspark_stage_identity.h"

namespace pih {

inline constexpr char kDeepSeekDsparkPrefillReferenceAbi[] =
    "deepseek_dspark_prefill_reference_fixture_v1";

// Reduced deterministic CPU receipt for the official prefill-only edge.  It
// models stage-specific WKV/KV-norm/RoPE and recent-ring placement, but it is
// not official-checkpoint or target-GPU numerical evidence.
struct DeepSeekDsparkPrefillReferenceReceipt final {
  static constexpr std::uint32_t kWindowSize = 4;
  static constexpr std::uint32_t kMainHiddenSize = 3;
  static constexpr std::uint32_t kKvSize = 4;
  static constexpr std::uint32_t kPromptTokens = 7;

  std::array<std::vector<BFloat16>, kDeepSeekDsparkStageCount>
      physical_recent_state;
  std::array<std::vector<BFloat16>, kDeepSeekDsparkStageCount>
      logical_recent_state;
  std::array<std::uint32_t, kWindowSize> logical_positions{};
  std::array<std::uint32_t, kDeepSeekDsparkStageCount> write_counts{};
};

Result<DeepSeekDsparkPrefillReferenceReceipt>
build_deepseek_dspark_prefill_reference(
    std::span<const std::uint32_t> chunk_sizes = {});

}  // namespace pih
