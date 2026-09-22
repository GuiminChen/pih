#pragma once

#include <vector>

#include "pih/model/deepseek_rank_compute_work_builder.h"

namespace pih {

struct DeepSeekDenseAttentionLayerPlanInput final {
  std::uint32_t layer = 0;
  std::vector<DeepSeekDenseAttentionStageSequenceWork> sequences;
};

struct DeepSeekMhcLayerPlanInput final {
  std::uint32_t layer = 0;
  std::vector<DeepSeekMhcStageSequenceWork> sequences;
};

class DeepSeekDenseMhcWorkFactory final {
 public:
  static Result<DeepSeekDenseMhcWorkFactory> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens);

  Status append_plan_work(
      std::uint32_t token_count, std::uint32_t sequence_count,
      std::vector<DeepSeekDenseAttentionLayerPlanInput> dense,
      std::vector<DeepSeekMhcLayerPlanInput> mhc_attention,
      std::vector<DeepSeekMhcLayerPlanInput> mhc_feed_forward,
      DeepSeekRankComputeWorkBuilder& builder) const;
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::uint32_t maximum_sequences_ = 0;
  std::uint32_t maximum_tokens_ = 0;
};

}  // namespace pih
