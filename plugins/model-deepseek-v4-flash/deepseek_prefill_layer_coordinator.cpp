#include "pih/model/deepseek_prefill_layer_coordinator.h"

#include <vector>

#include "pih/model/deepseek_compressed_page_mutation_assembler.h"

namespace pih {

Result<DeepSeekPrefillLayerCoordinator>
DeepSeekPrefillLayerCoordinator::Create(
    DeepSeekRecentStateWriter& recent_writer,
    DeepSeekCompressedLayerUpdateCoordinator& update_coordinator,
    DeepSeekIndexSelectionDriver& selection,
    DeepSeekSparseAttentionDriver& sparse_attention,
    DeepSeekAttentionLayerCoordinator& attention_coordinator) {
  DeepSeekPrefillLayerCoordinator coordinator;
  coordinator.recent_writer_ = &recent_writer;
  coordinator.update_coordinator_ = &update_coordinator;
  coordinator.selection_ = &selection;
  coordinator.sparse_attention_ = &sparse_attention;
  coordinator.attention_coordinator_ = &attention_coordinator;
  return coordinator;
}

Status DeepSeekPrefillLayerCoordinator::launch(
    const DeepSeekPrefillLayerSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  const auto count = submission.recent.size();
  if (submission.attention.kind == DeepSeekCompressedAttentionKind::kRecentOnly) {
    if (poisoned_ || count != 1 || !submission.updates.empty() ||
        submission.attention.query_positions.size() != 1 ||
        submission.recent.front().layer_id >= 2 ||
        submission.attention.query_positions.front() !=
            submission.recent.front().absolute_position ||
        submission.attention.compressed_slot_count != 0 ||
        submission.attention.has_indexer_projection) {
      return Status::InvalidArgument(
          "DeepSeek recent-only prefill requires one sequential token");
    }
    auto status = recent_writer_->launch(submission.recent.front(), transaction);
    if (status.ok()) status = attention_coordinator_->begin(submission.attention, transaction);
    if (!status.ok()) poisoned_ = true;
    return status;
  }
  if (poisoned_ ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      count == 0 || count > 4096 || submission.updates.size() != count ||
      submission.attention.query_positions.size() != count ||
      submission.attention.attention.index_matrix != nullptr ||
      submission.attention.attention.stream != transaction.stream()) {
    return Status::FailedPrecondition(
        "DeepSeek prefill layer chunk is not launchable");
  }

  const auto ratio = submission.updates.front().ratio;
  if (ratio != 4 && ratio != 128) {
    return Status::InvalidArgument("DeepSeek prefill ratio is invalid");
  }
  const auto layer = submission.updates.front().main_state.layer_id;
  const bool deferred = submission.updates.front().defer_page_mutation;
  const auto deferred_page = submission.updates.front().deferred_page;
  for (std::size_t token = 0; token < count; ++token) {
    const auto position = submission.recent[token].absolute_position;
    if (submission.recent[token].layer_id != layer ||
        submission.updates[token].ratio != ratio ||
        submission.updates[token].main_state.layer_id != layer ||
        submission.updates[token].main_state.absolute_position != position ||
        submission.updates[token].defer_page_mutation != deferred ||
        (deferred &&
         (submission.updates[token].deferred_page.main_rms_weight_bf16 !=
              deferred_page.main_rms_weight_bf16 ||
          submission.updates[token].deferred_page.index_rms_weight_bf16 !=
              deferred_page.index_rms_weight_bf16 ||
          submission.updates[token].deferred_page.main_cos_sin_cache_f32 !=
              deferred_page.main_cos_sin_cache_f32 ||
          submission.updates[token].deferred_page.index_cos_sin_cache_f32 !=
              deferred_page.index_cos_sin_cache_f32 ||
          submission.updates[token].deferred_page.rms_epsilon !=
              deferred_page.rms_epsilon)) ||
        submission.attention.query_positions[token] != position ||
        (token != 0 && position !=
                           submission.recent[token - 1].absolute_position + 1U)) {
      return Status::InvalidArgument(
          "DeepSeek prefill token sequence is inconsistent");
    }
    auto status = recent_writer_->validate(submission.recent[token],
                                           transaction);
    if (!status.ok()) return status;
    status = update_coordinator_->validate(submission.updates[token],
                                           transaction);
    if (!status.ok()) return status;
  }

  std::vector<DeepSeekCompressedLayerUpdateSubmission> bound_updates;
  std::span<const DeepSeekCompressedLayerUpdateSubmission> updates =
      submission.updates;
  if (deferred) {
    bound_updates.assign(submission.updates.begin(), submission.updates.end());
    for (auto& update : bound_updates) update.defer_page_mutation = false;
    auto bound = DeepSeekCompressedPageMutationAssembler::BindChunk(
        std::move(bound_updates), transaction,
        deferred_page.main_rms_weight_bf16,
        deferred_page.index_rms_weight_bf16,
        deferred_page.main_cos_sin_cache_f32,
        deferred_page.index_cos_sin_cache_f32,
        deferred_page.rms_epsilon);
    if (!bound.ok()) return bound.status();
    bound_updates = std::move(*bound);
    updates = bound_updates;
    for (const auto& update : updates) {
      auto status = update_coordinator_->validate(update, transaction);
      if (!status.ok()) return status;
    }
  }

  const auto final_position = submission.recent.back().absolute_position;
  const auto expected_slots = (final_position + 1U) / ratio;
  const auto expected_kind =
      ratio == 4 ? DeepSeekCompressedAttentionKind::kRatio4
                 : DeepSeekCompressedAttentionKind::kRatio128;
  if (submission.attention.kind != expected_kind ||
      submission.attention.compressed_slot_count != expected_slots) {
    return Status::InvalidArgument(
        "DeepSeek prefill compressed coverage is inconsistent");
  }

  Result<DeepSeekSparseIndexMatrix> shape =
      Status::Internal("DeepSeek prefill index shape was not resolved");
  if (ratio == 4 && expected_slots != 0) {
    const auto& selection = submission.attention.selection;
    if (selection.query_count != count ||
        selection.total_slot_count != expected_slots ||
        selection.stream != transaction.stream() ||
        selection.visible_slot_counts.size() != count) {
      return Status::InvalidArgument(
          "DeepSeek prefill selection shape is inconsistent");
    }
    for (std::size_t token = 0; token < count; ++token) {
      if (selection.visible_slot_counts[token] !=
          (submission.recent[token].absolute_position + 1U) / 4U) {
        return Status::InvalidArgument(
            "DeepSeek prefill selection visibility is inconsistent");
      }
    }
    auto status = selection_->validate(selection);
    if (!status.ok()) return status;
    const std::vector<std::vector<std::uint32_t>> empty(count);
    shape = DeepSeekAttentionIndexAssembler::Ratio4(
        submission.attention.query_positions, empty, expected_slots,
        submission.attention.recent_physical_offset,
        submission.attention.compressed_physical_offset);
  } else if (ratio == 4) {
    const std::vector<std::vector<std::uint32_t>> empty(count);
    shape = DeepSeekAttentionIndexAssembler::Ratio4(
        submission.attention.query_positions, empty, 0,
        submission.attention.recent_physical_offset,
        submission.attention.compressed_physical_offset);
  } else {
    shape = DeepSeekAttentionIndexAssembler::Ratio128(
        submission.attention.query_positions, expected_slots,
        submission.attention.recent_physical_offset,
        submission.attention.compressed_physical_offset);
  }
  if (!shape.ok()) return shape.status();
  auto sparse = submission.attention.attention;
  auto status = sparse_attention_->validate(sparse, *shape, transaction);
  if (!status.ok()) return status;

  auto run = [this](Status value) {
    if (!value.ok()) poisoned_ = true;
    return value;
  };
  for (std::size_t token = 0; token < count; ++token) {
    status = run(recent_writer_->launch(submission.recent[token], transaction));
    if (!status.ok()) return status;
    status = run(update_coordinator_->launch(updates[token],
                                             transaction));
    if (!status.ok()) return status;
  }
  return run(attention_coordinator_->begin(submission.attention, transaction));
}

}  // namespace pih
