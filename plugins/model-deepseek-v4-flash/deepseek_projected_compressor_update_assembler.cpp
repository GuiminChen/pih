#include "pih/model/deepseek_projected_compressor_update_assembler.h"

#include <limits>

namespace pih { namespace {

Result<std::uintptr_t> ape_row(std::uintptr_t base, std::uint32_t position,
                               std::uint32_t ratio,
                               std::uint32_t projection_width) {
  const auto row = static_cast<std::uint64_t>(position % ratio);
  const auto offset = row * projection_width * sizeof(float);
  if (base == 0 || offset >
          std::numeric_limits<std::uintptr_t>::max() - base) {
    return Status::InvalidArgument(
        "DeepSeek compressor APE row address is invalid");
  }
  return base + static_cast<std::uintptr_t>(offset);
}

bool main_complete(const DeepSeekCompressorWeightBindings& weights,
                   const DeepSeekCompressorProjectionSlice& scratch) {
  return weights.main_wkv_bf16 != 0 && weights.main_wgate_bf16 != 0 &&
         weights.main_ape_f32 != 0 && weights.main_norm_bf16 != 0 &&
         weights.generation != 0 && scratch.main_kv_f32 != 0 &&
         scratch.main_gate_f32 != 0 && scratch.main_output_f32 != 0;
}

bool index_complete(const DeepSeekCompressorWeightBindings& weights,
                    const DeepSeekCompressorProjectionSlice& scratch) {
  return weights.indexer_wq_b_e4m3 != 0 &&
         weights.indexer_wq_b_scale_bits != 0 &&
         weights.indexer_weights_proj_bf16 != 0 &&
         weights.indexer_wkv_bf16 != 0 && weights.indexer_wgate_bf16 != 0 &&
         weights.indexer_ape_f32 != 0 && weights.indexer_norm_bf16 != 0 &&
         scratch.index_kv_f32 != 0 && scratch.index_gate_f32 != 0 &&
         scratch.index_output_f32 != 0 && scratch.index_weights_f32 != 0;
}

DeepSeekCompressorStateSubmission state(
    std::uint32_t layer, bool indexer, std::uintptr_t hidden,
    std::uintptr_t kv_weight, std::uintptr_t gate_weight, std::uintptr_t kv, std::uintptr_t gate,
    std::uintptr_t ape, std::uintptr_t output, std::uintptr_t error,
    std::uintptr_t stream, std::uint32_t position, std::uint32_t ratio,
    std::uint32_t head_dim) {
  DeepSeekCompressorStateSubmission result{
      layer, indexer, kv, gate, ape, output, error, stream, 1, position};
  result.projection = {hidden, kv_weight, gate_weight, kv, gate, error, stream,
                       1, ratio, head_dim, 4096};
  return result;
}

}  // namespace

Result<DeepSeekCompressedLayerUpdateSubmission>
DeepSeekProjectedCompressorUpdateAssembler::AssembleDecode(
    std::uint32_t layer, DeepSeekCompressedAttentionKind kind,
    std::uintptr_t hidden_row_bf16,
    const DeepSeekCompressorWeightBindings& weights,
    const DeepSeekCompressorProjectionSlice& scratch,
    std::uintptr_t device_error_u32, std::uintptr_t stream,
    std::uint32_t absolute_position) {
  if (layer > 42 || hidden_row_bf16 == 0 || device_error_u32 == 0 ||
      stream == 0 || absolute_position >= 1048576 ||
      !main_complete(weights, scratch)) {
    return Status::InvalidArgument(
        "DeepSeek projected compressor update input is invalid");
  }
  DeepSeekCompressedLayerUpdateSubmission result;
  if (kind == DeepSeekCompressedAttentionKind::kRatio4) {
    if (!index_complete(weights, scratch)) {
      return Status::InvalidArgument(
          "DeepSeek ratio-4 projected compressor indexer is incomplete");
    }
    auto main_ape = ape_row(weights.main_ape_f32, absolute_position, 4, 1024);
    if (!main_ape.ok()) return main_ape.status();
    auto index_ape = ape_row(weights.indexer_ape_f32,
                             absolute_position, 4, 256);
    if (!index_ape.ok()) return index_ape.status();
    result.ratio = 4;
    result.main_state = state(
        layer, false, hidden_row_bf16, weights.main_wkv_bf16, weights.main_wgate_bf16,
        scratch.main_kv_f32, scratch.main_gate_f32, *main_ape,
        scratch.main_output_f32, device_error_u32, stream,
        absolute_position, 4, 512);
    result.index_state = state(
        layer, true, hidden_row_bf16,
        weights.indexer_wkv_bf16, weights.indexer_wgate_bf16, scratch.index_kv_f32,
        scratch.index_gate_f32, *index_ape, scratch.index_output_f32,
        device_error_u32, stream, absolute_position, 4, 128);
    return result;
  }
  if (kind != DeepSeekCompressedAttentionKind::kRatio128) {
    return Status::InvalidArgument(
        "DeepSeek projected compressor kind is invalid");
  }
  auto main_ape = ape_row(weights.main_ape_f32, absolute_position, 128, 512);
  if (!main_ape.ok()) return main_ape.status();
  result.ratio = 128;
  result.main_state = state(
      layer, false, hidden_row_bf16, weights.main_wkv_bf16, weights.main_wgate_bf16,
      scratch.main_kv_f32, scratch.main_gate_f32, *main_ape,
      scratch.main_output_f32, device_error_u32, stream,
      absolute_position, 128, 512);
  return result;
}

}  // namespace pih
