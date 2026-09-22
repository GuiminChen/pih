#include "pih/model/deepseek_projected_attention_shape_assembler.h"

#include <algorithm>
#include <limits>

#include "pih/model/deepseek_projected_compressor_update_assembler.h"

namespace pih {

Status DeepSeekProjectedAttentionShapeAssembler::BindSparseRuntime(
    DeepSeekAttentionLayerPlanShapeSeed& shape,
    std::uintptr_t sparse_query_bf16, std::uintptr_t sparse_kv_bf16,
    std::uintptr_t sparse_output_bf16,
    std::uintptr_t attention_sink_f32,
    const DeepSeekAttentionDeviceScratchResources& attention_scratch) {
  if (shape.positions.empty() ||
      (shape.ratio != 0 && shape.ratio != 4 && shape.ratio != 128) ||
      shape.recent_physical_offset < 0 ||
      shape.compressed_physical_offset < 0 ||
      shape.positions.size() > attention_scratch.maximum_queries() ||
      sparse_query_bf16 == 0 || sparse_kv_bf16 == 0 ||
      sparse_output_bf16 == 0 || attention_sink_f32 == 0 ||
      attention_scratch.sparse_indices_i32() == 0 ||
      attention_scratch.sparse_page_slots_u32() == 0 ||
      attention_scratch.sparse_error_u32() == 0) {
    return Status::InvalidArgument(
        "DeepSeek sparse runtime binding is invalid");
  }
  shape.sparse_query_bf16 = sparse_query_bf16;
  shape.sparse_kv_bf16 = sparse_kv_bf16;
  shape.sparse_output_bf16 = sparse_output_bf16;
  shape.attention_sink_f32 = attention_sink_f32;
  shape.sparse_indices_i32 = attention_scratch.sparse_indices_i32();
  shape.sparse_page_slots_u32 = attention_scratch.sparse_page_slots_u32();
  shape.sparse_error_u32 = attention_scratch.sparse_error_u32();
  const auto compressed_slots = shape.ratio == 0 ? 0U :
      (static_cast<std::uint64_t>(shape.positions.back()) + 1U) / shape.ratio;
  const auto recent_end =
      static_cast<std::uint64_t>(shape.recent_physical_offset) + 128U;
  const auto compressed_end =
      static_cast<std::uint64_t>(shape.compressed_physical_offset) +
      compressed_slots;
  const auto logical_kv_count = std::max(recent_end, compressed_end);
  if (logical_kv_count == 0 || logical_kv_count > 1048704U) {
    return Status::InvalidArgument(
        "DeepSeek sparse logical KV address space is invalid");
  }
  shape.sparse_kv_count = static_cast<std::uint32_t>(logical_kv_count);
  return Status::Ok();
}

Status DeepSeekProjectedAttentionShapeAssembler::Populate(
    DeepSeekAttentionLayerPlanShapeSeed& shape,
    std::uintptr_t hidden_rows_bf16, std::uintptr_t qr_bf16,
    std::uintptr_t positions_u32,
    const DeepSeekCompressorWeightBindings& weights,
    const DeepSeekCompressorProjectionDeviceResources& scratch,
    const DeepSeekAttentionDeviceScratchResources& attention_scratch,
    std::uintptr_t compressor_error_u32, std::uintptr_t stream,
    std::uintptr_t cos_sin_cache_f32, std::uint32_t table_position_count) {
  constexpr std::uintptr_t kHiddenRowBytes = 4096U * sizeof(std::uint16_t);
  if (shape.layer > 42 || (shape.ratio != 4 && shape.ratio != 128) ||
      shape.positions.empty() || shape.positions.size() > 4096 ||
      !shape.updates.empty() || hidden_rows_bf16 == 0 || qr_bf16 == 0 ||
      positions_u32 == 0 || table_position_count == 0 || stream == 0 ||
      compressor_error_u32 == 0 ||
      cos_sin_cache_f32 == 0 ||
      shape.positions.size() > scratch.maximum_queries() ||
      shape.positions.size() > attention_scratch.maximum_queries()) {
    return Status::InvalidArgument(
        "DeepSeek projected attention shape input is invalid");
  }
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  updates.reserve(shape.positions.size());
  const auto kind = shape.ratio == 4
      ? DeepSeekCompressedAttentionKind::kRatio4
      : DeepSeekCompressedAttentionKind::kRatio128;
  for (std::size_t token = 0; token < shape.positions.size(); ++token) {
    if (token > (std::numeric_limits<std::uintptr_t>::max() -
                 hidden_rows_bf16) / kHiddenRowBytes) {
      return Status::InvalidArgument(
          "DeepSeek projected hidden row address overflows");
    }
    auto slice = scratch.slice(static_cast<std::uint32_t>(token));
    if (!slice.ok()) return slice.status();
    auto update = DeepSeekProjectedCompressorUpdateAssembler::AssembleDecode(
        shape.layer, kind, hidden_rows_bf16 + token * kHiddenRowBytes,
        weights, *slice, compressor_error_u32, stream,
        shape.positions[token]);
    if (!update.ok()) return update.status();
    updates.push_back(std::move(*update));
  }
  shape.updates = std::move(updates);
  shape.bind_compressed_page_mutations = true;
  shape.main_rms_weight_bf16 = weights.main_norm_bf16;
  shape.index_rms_weight_bf16 =
      shape.ratio == 4 ? weights.indexer_norm_bf16 : 0;
  shape.main_cos_sin_cache_f32 = cos_sin_cache_f32;
  shape.index_cos_sin_cache_f32 =
      shape.ratio == 4 ? cos_sin_cache_f32 : 0;
  shape.rms_epsilon = DeepSeekRmsNormLaunch::kEpsilon;
  // Ratio-4 plan validation requires its scratch bindings even before the
  // first compressed slot exists. Only the projection launch is conditional.
  if (shape.ratio == 4) {
    shape.index_query_bf16 = attention_scratch.indexer_query_bf16();
    shape.index_head_weight_f32 =
        attention_scratch.indexer_head_weight_f32();
    shape.index_arena = attention_scratch.index_arena();
  }
  if (shape.ratio == 4 && (shape.positions.back() + 1U) / 4U != 0) {
    shape.has_indexer_projection = true;
    shape.indexer_projection = {
        qr_bf16, hidden_rows_bf16, weights.indexer_wq_b_e4m3,
        weights.indexer_wq_b_scale_bits, attention_scratch.indexer_qr_e4m3(),
        attention_scratch.indexer_qr_scale_bits(),
        weights.indexer_weights_proj_bf16, cos_sin_cache_f32,
        positions_u32, shape.index_query_bf16,
        shape.index_head_weight_f32,
        attention_scratch.indexer_projection_error_u32(), stream,
        static_cast<std::uint32_t>(shape.positions.size()),
        table_position_count};
  }
  return Status::Ok();
}

}  // namespace pih
