#pragma once

#include <array>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"
#include "pih/model/deepseek_dspark_stage_identity.h"

namespace pih {

inline constexpr char kDeepSeekDsparkThreeStageReferenceAbi[] =
    "deepseek_dspark_three_stage_reference_fixture_v1";

// Reduced, deterministic CPU contract fixture. It proves stage ordering,
// boundary ownership and five-position causal head geometry; it is not a
// substitute for official-checkpoint or target-GPU numerical evidence.
struct DeepSeekDsparkThreeStageReferenceReceipt final {
  static constexpr std::uint32_t kBlockSize = 5;
  static constexpr std::uint32_t kHiddenSize = 4;
  static constexpr std::uint32_t kHcStreams = 4;
  static constexpr std::uint32_t kVocabularySize = 7;
  static constexpr std::uint32_t kMarkovRank = 2;

  std::array<std::vector<BFloat16>, kDeepSeekDsparkStageCount> stage_hidden_hc;
  std::vector<BFloat16> head_hidden;
  std::vector<float> raw_logits;
  std::vector<float> markov_bias;
  std::vector<float> biased_logits;
  std::array<std::uint32_t, kBlockSize> proposal_token_ids{};
  std::array<std::uint32_t, kBlockSize + 1> causal_token_chain{};
  std::vector<float> confidence;
};

Result<DeepSeekDsparkThreeStageReferenceReceipt>
build_deepseek_dspark_three_stage_reference();

}  // namespace pih
