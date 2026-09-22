#include "pih/model/deepseek_deferred_attention_shape_mutation_binder.h"

namespace pih {

Status DeepSeekDeferredAttentionShapeMutationBinder::Bind(
    std::uint32_t sequence,
    std::span<DeepSeekAttentionLayerPlanShapeSeed> shapes,
    DeepSeekAttentionSequenceTransaction& transaction,
    DeepSeekRatio4PagePool& ratio4_pool,
    DeepSeekRatio128PagePool& ratio128_pool) {
  if (sequence == 0 || shapes.empty() ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.sequence() != sequence) {
    return Status::InvalidArgument(
        "DeepSeek deferred attention mutation binding is invalid");
  }
  for (const auto& shape : shapes) {
    if (shape.ratio == 0 && shape.layer < 2 && shape.positions.size() == 1 &&
        shape.updates.empty() && !shape.bind_compressed_page_mutations) continue;
    if (!shape.bind_compressed_page_mutations || shape.layer >= 43 ||
        (shape.ratio != 4 && shape.ratio != 128) ||
        shape.positions.size() != 1 || shape.updates.size() != 1 ||
        shape.positions[0] != shape.updates[0].main_state.absolute_position ||
        shape.updates[0].main_state.layer_id != shape.layer) {
      return Status::InvalidArgument(
          "DeepSeek deferred decode mutation shape is invalid");
    }
  }
  for (auto& shape : shapes) {
    if (shape.ratio == 0) continue;
    const auto position = shape.positions[0];
    const auto stored_slot = position / shape.ratio;
    const bool boundary = (position + 1U) % shape.ratio == 0;
    const bool tail = boundary && stored_slot % 64U != 0;
    const auto logical_page = stored_slot / 64U;
    std::optional<DeepSeekRatio4PagePair> committed4;
    std::optional<DeepSeekRatio128Page> committed128;
    if (tail && shape.ratio == 4) {
      auto found = ratio4_pool.find_published(
          sequence, shape.layer, logical_page);
      if (!found.ok()) return found.status();
      if (!found->has_value()) {
        return Status::FailedPrecondition(
            "DeepSeek ratio-4 published tail is missing");
      }
      committed4 = **found;
    } else if (tail) {
      auto found = ratio128_pool.find_published(
          sequence, shape.layer, logical_page);
      if (!found.ok()) return found.status();
      if (!found->has_value()) {
        return Status::FailedPrecondition(
            "DeepSeek ratio-128 published tail is missing");
      }
      committed128 = **found;
    }
    auto bound = DeepSeekCompressedPageMutationAssembler::BindDecode(
        shape.updates[0], transaction, shape.main_rms_weight_bf16,
        shape.index_rms_weight_bf16, shape.main_cos_sin_cache_f32,
        shape.index_cos_sin_cache_f32, shape.rms_epsilon,
        committed4.has_value() ? &*committed4 : nullptr,
        committed128.has_value() ? &*committed128 : nullptr);
    if (!bound.ok()) {
      const auto aborted = transaction.cancel_preparing();
      return aborted.ok() ? bound.status() : aborted;
    }
    shape.updates[0] = std::move(*bound);
    shape.bind_compressed_page_mutations = false;
  }
  return Status::Ok();
}

}  // namespace pih
