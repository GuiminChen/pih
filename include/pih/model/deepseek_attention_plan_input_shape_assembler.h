#pragma once

#include <vector>

#include "pih/model/deepseek_attention_work_factory.h"

namespace pih {

struct DeepSeekAttentionLayerPlanShapeSeed final {
  std::uint32_t layer = 0;
  std::uint32_t ratio = 0;
  std::vector<std::uint32_t> positions;
  std::vector<DeepSeekRecentStateSubmission> recent;
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  std::uintptr_t index_query_bf16 = 0;
  std::uintptr_t index_kv_bf16 = 0;
  std::uintptr_t index_head_weight_f32 = 0;
  bool has_indexer_projection = false;
  DeepSeekIndexerProjectionLaunch indexer_projection;
  DeepSeekIndexSelectionArena index_arena;
  std::uintptr_t sparse_query_bf16 = 0;
  std::uintptr_t sparse_kv_bf16 = 0;
  std::uintptr_t attention_sink_f32 = 0;
  std::uintptr_t sparse_indices_i32 = 0;
  std::uintptr_t sparse_page_slots_u32 = 0;
  std::uintptr_t sparse_output_bf16 = 0;
  std::uintptr_t sparse_error_u32 = 0;
  std::uint32_t sparse_kv_count = 0;
  std::int32_t recent_physical_offset = 0;
  std::int32_t compressed_physical_offset = 0;
  std::uintptr_t stream = 0;
  bool bind_compressed_page_mutations = false;
  std::uintptr_t main_rms_weight_bf16 = 0;
  std::uintptr_t index_rms_weight_bf16 = 0;
  std::uintptr_t main_cos_sin_cache_f32 = 0;
  std::uintptr_t index_cos_sin_cache_f32 = 0;
  float rms_epsilon = 0.0F;
};

class DeepSeekOwnedAttentionLayerPlanInput final {
 public:
  static Result<DeepSeekOwnedAttentionLayerPlanInput> AssembleDecode(
      DeepSeekAttentionLayerPlanShapeSeed seed);
  static Result<DeepSeekOwnedAttentionLayerPlanInput> AssembleChunk(
      DeepSeekAttentionLayerPlanShapeSeed seed);

  DeepSeekOwnedAttentionLayerPlanInput(
      DeepSeekOwnedAttentionLayerPlanInput&& other) noexcept;
  DeepSeekOwnedAttentionLayerPlanInput& operator=(
      DeepSeekOwnedAttentionLayerPlanInput&& other) noexcept;
  DeepSeekOwnedAttentionLayerPlanInput(
      const DeepSeekOwnedAttentionLayerPlanInput&) = delete;
  DeepSeekOwnedAttentionLayerPlanInput& operator=(
      const DeepSeekOwnedAttentionLayerPlanInput&) = delete;

  [[nodiscard]] const DeepSeekDecodeAttentionLayerPlanInput& decode() const {
    return decode_;
  }
  [[nodiscard]] const DeepSeekChunkAttentionLayerPlanInput& chunk() const {
    return chunk_;
  }
  [[nodiscard]] bool is_decode() const noexcept { return decode_mode_; }

 private:
  DeepSeekOwnedAttentionLayerPlanInput() = default;
  void rebind_views() noexcept;

  bool decode_mode_ = false;
  std::vector<std::uint32_t> positions_;
  std::vector<std::uint32_t> visible_slot_counts_;
  std::vector<DeepSeekRecentStateSubmission> recent_;
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates_;
  DeepSeekDecodeAttentionLayerPlanInput decode_;
  DeepSeekChunkAttentionLayerPlanInput chunk_;
};

}  // namespace pih
