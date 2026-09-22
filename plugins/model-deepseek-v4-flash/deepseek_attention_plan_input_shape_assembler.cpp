#include "pih/model/deepseek_attention_plan_input_shape_assembler.h"

#include <cmath>
#include <limits>

namespace pih {
namespace {

Status validate_seed(const DeepSeekAttentionLayerPlanShapeSeed& seed,
                     bool decode) {
  const bool recent_only = seed.ratio == 0;
  if (seed.layer > 42 ||
      (recent_only ? seed.layer >= 2 : (seed.ratio != 4 && seed.ratio != 128)) ||
      seed.positions.empty() || seed.positions.size() > 4096 ||
      (decode && seed.positions.size() != 1) ||
      seed.recent.size() != seed.positions.size() ||
      (recent_only ? (!seed.updates.empty() || seed.bind_compressed_page_mutations)
                   : seed.updates.size() != seed.positions.size()) || seed.stream == 0 ||
      seed.sparse_query_bf16 == 0 || seed.sparse_kv_bf16 == 0 ||
      seed.attention_sink_f32 == 0 || seed.sparse_indices_i32 == 0 ||
      seed.sparse_page_slots_u32 == 0 ||
      seed.sparse_output_bf16 == 0 || seed.sparse_error_u32 == 0 ||
      seed.sparse_kv_count == 0) {
    return Status::InvalidArgument(
        "DeepSeek scheduler attention shape seed is invalid");
  }
  if (seed.ratio == 4 &&
      (seed.index_query_bf16 == 0 || seed.index_kv_bf16 == 0 ||
       seed.index_head_weight_f32 == 0 ||
       seed.index_arena.score_f32 == 0 ||
       seed.index_arena.error_flag_u32 == 0)) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 scheduler index resources are invalid");
  }
  if (seed.bind_compressed_page_mutations) {
    if (seed.main_rms_weight_bf16 == 0 ||
        seed.main_cos_sin_cache_f32 == 0 ||
        !std::isfinite(seed.rms_epsilon) || seed.rms_epsilon <= 0.0F ||
        (seed.ratio == 4 &&
         (seed.index_rms_weight_bf16 == 0 ||
          seed.index_cos_sin_cache_f32 == 0)) ||
        (seed.ratio == 128 &&
         (seed.index_rms_weight_bf16 != 0 ||
          seed.index_cos_sin_cache_f32 != 0))) {
      return Status::InvalidArgument(
          "DeepSeek deferred compressed page resources are invalid");
    }
    for (const auto& update : seed.updates) {
      if (update.has_completed_slot || update.defer_page_mutation) {
        return Status::InvalidArgument(
            "DeepSeek compressed page mutation is already bound");
      }
    }
  }
  const auto slots = recent_only ? 0U : (seed.positions.back() + 1U) / seed.ratio;
  if ((seed.ratio == 4 && slots != 0) != seed.has_indexer_projection ||
      (seed.has_indexer_projection &&
       (seed.indexer_projection.token_count != seed.positions.size() ||
        seed.indexer_projection.query_bf16 != seed.index_query_bf16 ||
        seed.indexer_projection.head_weight_f32 !=
            seed.index_head_weight_f32 ||
        seed.indexer_projection.error_flag_u32 == 0 ||
        seed.indexer_projection.error_flag_u32 ==
            seed.index_arena.error_flag_u32 ||
        seed.indexer_projection.stream != seed.stream))) {
    return Status::InvalidArgument(
        "DeepSeek scheduler indexer projection is inconsistent");
  }
  for (std::size_t token = 0; token < seed.positions.size(); ++token) {
    const auto position = seed.positions[token];
    if ((token != 0 &&
         (seed.positions[token - 1] ==
              std::numeric_limits<std::uint32_t>::max() ||
          position != seed.positions[token - 1] + 1U)) ||
        position >= 1048576 || seed.recent[token].layer_id != seed.layer ||
        seed.recent[token].absolute_position != position ||
        seed.recent[token].stream != seed.stream ||
        seed.recent[token].batch_count != 1 ||
        (!recent_only && (seed.updates[token].ratio != seed.ratio ||
        seed.updates[token].main_state.layer_id != seed.layer ||
        seed.updates[token].main_state.absolute_position != position ||
        seed.updates[token].main_state.stream != seed.stream))) {
      return Status::InvalidArgument(
          "DeepSeek scheduler attention token sequence is inconsistent");
    }
  }
  return Status::Ok();
}

DeepSeekAttentionLayerSubmission attention_submission(
    const DeepSeekAttentionLayerPlanShapeSeed& seed,
    std::span<const std::uint32_t> positions,
    std::span<const std::uint32_t> visible) {
  const auto slots = seed.ratio == 0 ? 0U : (positions.back() + 1U) / seed.ratio;
  DeepSeekAttentionLayerSubmission result;
  result.kind = seed.ratio == 0 ? DeepSeekCompressedAttentionKind::kRecentOnly
      : seed.ratio == 4 ? DeepSeekCompressedAttentionKind::kRatio4
                                : DeepSeekCompressedAttentionKind::kRatio128;
  result.query_positions = positions;
  result.compressed_slot_count = slots;
  result.recent_physical_offset = seed.recent_physical_offset;
  result.compressed_physical_offset = seed.compressed_physical_offset;
  if (seed.ratio == 4 && slots != 0) {
    result.has_indexer_projection = seed.has_indexer_projection;
    result.indexer_projection = seed.indexer_projection;
    result.selection = {
        seed.index_query_bf16, seed.index_kv_bf16,
        seed.index_head_weight_f32, seed.index_arena, visible, slots,
        static_cast<std::uint32_t>(positions.size()), 64, seed.stream};
  }
  result.attention = {
      nullptr, seed.sparse_query_bf16, seed.sparse_kv_bf16,
      seed.attention_sink_f32, seed.sparse_indices_i32,
      seed.sparse_output_bf16, seed.sparse_error_u32, seed.stream, 64,
      seed.sparse_kv_count};
  result.attention.device_page_slots_u32 = seed.sparse_page_slots_u32;
  return result;
}

}  // namespace

Result<DeepSeekOwnedAttentionLayerPlanInput>
DeepSeekOwnedAttentionLayerPlanInput::AssembleDecode(
    DeepSeekAttentionLayerPlanShapeSeed seed) {
  auto status = validate_seed(seed, true);
  if (!status.ok()) return status;
  if (seed.bind_compressed_page_mutations) {
    auto& update = seed.updates.front();
    update.defer_page_mutation = true;
    update.deferred_page = {
        seed.main_rms_weight_bf16, seed.index_rms_weight_bf16,
        seed.main_cos_sin_cache_f32, seed.index_cos_sin_cache_f32,
        seed.rms_epsilon};
  }
  DeepSeekOwnedAttentionLayerPlanInput result;
  result.decode_mode_ = true;
  result.positions_ = std::move(seed.positions);
  result.recent_ = std::move(seed.recent);
  result.updates_ = std::move(seed.updates);
  if (seed.ratio == 4) {
    result.visible_slot_counts_.push_back(
        (result.positions_.front() + 1U) / 4U);
  }
  result.decode_.layer = seed.layer;
  result.decode_.recent = result.recent_.front();
  if (!result.updates_.empty()) result.decode_.update = result.updates_.front();
  result.decode_.attention = attention_submission(
      seed, result.positions_, result.visible_slot_counts_);
  result.rebind_views();
  return result;
}

Result<DeepSeekOwnedAttentionLayerPlanInput>
DeepSeekOwnedAttentionLayerPlanInput::AssembleChunk(
    DeepSeekAttentionLayerPlanShapeSeed seed) {
  auto status = validate_seed(seed, false);
  if (!status.ok()) return status;
  if (seed.bind_compressed_page_mutations) {
    for (auto& update : seed.updates) {
      update.defer_page_mutation = true;
      update.deferred_page = {
          seed.main_rms_weight_bf16, seed.index_rms_weight_bf16,
          seed.main_cos_sin_cache_f32, seed.index_cos_sin_cache_f32,
          seed.rms_epsilon};
    }
  }
  DeepSeekOwnedAttentionLayerPlanInput result;
  result.positions_ = std::move(seed.positions);
  result.recent_ = std::move(seed.recent);
  result.updates_ = std::move(seed.updates);
  if (seed.ratio == 4) {
    result.visible_slot_counts_.reserve(result.positions_.size());
    for (const auto position : result.positions_) {
      result.visible_slot_counts_.push_back((position + 1U) / 4U);
    }
  }
  result.chunk_.layer = seed.layer;
  result.chunk_.submission.attention = attention_submission(
      seed, result.positions_, result.visible_slot_counts_);
  result.rebind_views();
  return result;
}

DeepSeekOwnedAttentionLayerPlanInput::DeepSeekOwnedAttentionLayerPlanInput(
    DeepSeekOwnedAttentionLayerPlanInput&& other) noexcept
    : decode_mode_(other.decode_mode_),
      positions_(std::move(other.positions_)),
      visible_slot_counts_(std::move(other.visible_slot_counts_)),
      recent_(std::move(other.recent_)), updates_(std::move(other.updates_)),
      decode_(other.decode_), chunk_(other.chunk_) {
  rebind_views();
}

DeepSeekOwnedAttentionLayerPlanInput&
DeepSeekOwnedAttentionLayerPlanInput::operator=(
    DeepSeekOwnedAttentionLayerPlanInput&& other) noexcept {
  if (this == &other) return *this;
  decode_mode_ = other.decode_mode_;
  positions_ = std::move(other.positions_);
  visible_slot_counts_ = std::move(other.visible_slot_counts_);
  recent_ = std::move(other.recent_);
  updates_ = std::move(other.updates_);
  decode_ = other.decode_;
  chunk_ = other.chunk_;
  rebind_views();
  return *this;
}

void DeepSeekOwnedAttentionLayerPlanInput::rebind_views() noexcept {
  if (decode_mode_) {
    decode_.attention.query_positions = positions_;
    decode_.attention.selection.visible_slot_counts = visible_slot_counts_;
  } else {
    chunk_.submission.recent = recent_;
    chunk_.submission.updates = updates_;
    chunk_.submission.attention.query_positions = positions_;
    chunk_.submission.attention.selection.visible_slot_counts =
        visible_slot_counts_;
  }
}

}  // namespace pih
