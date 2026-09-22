#pragma once

#include <vector>

#include "pih/model/deepseek_compressed_layer_update_coordinator.h"

namespace pih {

class DeepSeekCompressedPageMutationAssembler final {
 public:
  static Result<DeepSeekCompressedLayerUpdateSubmission> BindDecode(
      DeepSeekCompressedLayerUpdateSubmission update,
      DeepSeekAttentionSequenceTransaction& transaction,
      std::uintptr_t main_rms_weight_bf16,
      std::uintptr_t index_rms_weight_bf16,
      std::uintptr_t main_cos_sin_cache_f32,
      std::uintptr_t index_cos_sin_cache_f32,
      float rms_epsilon,
      const DeepSeekRatio4PagePair* committed_ratio4 = nullptr,
      const DeepSeekRatio128Page* committed_ratio128 = nullptr);

  static Result<std::vector<DeepSeekCompressedLayerUpdateSubmission>>
  BindChunk(
      std::vector<DeepSeekCompressedLayerUpdateSubmission> updates,
      DeepSeekAttentionSequenceTransaction& transaction,
      std::uintptr_t main_rms_weight_bf16,
      std::uintptr_t index_rms_weight_bf16,
      std::uintptr_t main_cos_sin_cache_f32,
      std::uintptr_t index_cos_sin_cache_f32,
      float rms_epsilon);
};

}  // namespace pih
