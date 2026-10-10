#pragma once

#include <array>
#include <optional>
#include <vector>

#include "pih/model/deepseek_rank_compute_work_builder.h"

namespace pih {

struct DeepSeekAttentionLayerRuntimeResources final {
  std::uint32_t layer = 0;
  DeepSeekRecentStateWriter* recent_writer = nullptr;
  DeepSeekCompressedLayerUpdateCoordinator* update_coordinator = nullptr;
  DeepSeekAttentionLayerCoordinator* attention_coordinator = nullptr;
  DeepSeekPrefillLayerCoordinator* prefill_coordinator = nullptr;
  std::uint32_t sequence_index = 0;
};

struct DeepSeekDecodeAttentionLayerPlanInput final {
  std::uint32_t layer = 0;
  DeepSeekRecentStateSubmission recent;
  DeepSeekCompressedLayerUpdateSubmission update;
  DeepSeekAttentionLayerSubmission attention;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  std::uint32_t sequence_index = 0;
};

struct DeepSeekChunkAttentionLayerPlanInput final {
  std::uint32_t layer = 0;
  DeepSeekPrefillLayerSubmission submission;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  std::uint32_t sequence_index = 0;
};

class DeepSeekAttentionWorkFactory final {
 public:
  static Result<DeepSeekAttentionWorkFactory> Create(
      DeepSeekStageRange owned_layers,
      std::vector<DeepSeekAttentionLayerRuntimeResources> layers);

  Status append_decode_plan_work(
      std::vector<DeepSeekDecodeAttentionLayerPlanInput> layers,
      DeepSeekRankComputeWorkBuilder& builder) const;
  Status append_chunk_plan_work(
      std::vector<DeepSeekChunkAttentionLayerPlanInput> layers,
      DeepSeekRankComputeWorkBuilder& builder) const;
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::array<std::vector<DeepSeekAttentionLayerRuntimeResources>, 43>
      layers_;
};

}  // namespace pih
