#include "pih/model/deepseek_attention_layer_coordinator.h"

namespace pih {

Result<DeepSeekAttentionLayerCoordinator>
DeepSeekAttentionLayerCoordinator::Create(
    DeepSeekIndexerProjectionCoordinator& indexer_projection,
    DeepSeekIndexSelectionDriver& selection,
    DeepSeekSparseAttentionDriver& attention) {
  DeepSeekAttentionLayerCoordinator coordinator;
  coordinator.indexer_projection_ = &indexer_projection;
  coordinator.selection_ = &selection;
  coordinator.attention_ = &attention;
  return coordinator;
}

Result<DeepSeekAttentionLayerCoordinator>
DeepSeekAttentionLayerCoordinator::CreatePaged(
    std::uint32_t layer_id, DeepSeekFixedStateLayout fixed_layout,
    DeepSeekAttentionPageArena page_arena,
    DeepSeekIndexerProjectionCoordinator& indexer_projection,
    DeepSeekIndexSelectionDriver& selection,
    DeepSeekSparseAttentionDriver& attention) {
  if (layer_id >= 43) {
    return Status::InvalidArgument(
        "DeepSeek paged attention layer identity is invalid");
  }
  auto created = Create(indexer_projection, selection, attention);
  if (!created.ok()) return created.status();
  created->layer_id_ = layer_id;
  created->paged_selection_ = true;
  created->fixed_layout_ = std::move(fixed_layout);
  created->page_arena_ = std::move(page_arena);
  return created;
}

Status DeepSeekAttentionLayerCoordinator::begin(
    const DeepSeekAttentionLayerSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (state_ != DeepSeekAttentionLayerCoordinatorState::kIdle ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      submission.query_positions.empty() ||
      submission.query_positions.size() > 4096 ||
      (submission.kind != DeepSeekCompressedAttentionKind::kRecentOnly &&
       submission.kind != DeepSeekCompressedAttentionKind::kRatio4 &&
       submission.kind != DeepSeekCompressedAttentionKind::kRatio128) ||
      (submission.kind == DeepSeekCompressedAttentionKind::kRecentOnly &&
       (submission.compressed_slot_count != 0 || submission.has_indexer_projection ||
        (paged_selection_ && layer_id_ >= 2))) ||
      submission.attention.index_matrix != nullptr ||
      submission.attention.stream != transaction.stream()) {
    return Status::FailedPrecondition(
        "DeepSeek attention layer coordinator is not beginable");
  }
  query_positions_.assign(submission.query_positions.begin(),
                          submission.query_positions.end());
  submission_ = submission;
  submission_.query_positions = query_positions_;
  transaction_ = &transaction;
  if (submission.kind == DeepSeekCompressedAttentionKind::kRatio4 &&
      submission.compressed_slot_count != 0) {
    if (!submission.has_indexer_projection ||
        submission.indexer_projection.token_count != query_positions_.size() ||
        submission.indexer_projection.query_bf16 !=
            submission.selection.query_bf16 ||
        submission.indexer_projection.head_weight_f32 !=
            submission.selection.head_weight_f32 ||
        submission.indexer_projection.stream != transaction.stream() ||
        submission.selection.query_count != query_positions_.size() ||
        submission.selection.total_slot_count !=
            submission.compressed_slot_count ||
        submission.selection.stream != transaction.stream()) {
      state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
      return Status::InvalidArgument(
          "DeepSeek ratio-4 projection and selection do not match the layer");
    }
    auto status = indexer_projection_->launch(
        submission.indexer_projection, transaction);
    if (!status.ok()) {
      state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
      return status;
    }
    if (paged_selection_) {
      const auto logical_pages =
          (submission.compressed_slot_count + 63U) / 64U;
      auto page_slots = transaction.ratio4_index_page_slots(
          layer_id_, logical_pages);
      if (!page_slots.ok()) {
        state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
        return page_slots.status();
      }
      status = selection_->begin_paged(
          submission.selection, *page_slots,
          transaction.ratio4_physical_page_count());
    } else {
      status = selection_->begin(submission.selection);
    }
    if (!status.ok()) {
      state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
      return status;
    }
    state_ = DeepSeekAttentionLayerCoordinatorState::kSelectionReady;
    return Status::Ok();
  }
  if (submission.has_indexer_projection) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return Status::InvalidArgument(
        "DeepSeek indexer projection is not allowed for this layer");
  }
  const std::vector<std::vector<std::uint32_t>> empty(query_positions_.size());
  return assemble_and_launch(empty);
}

Status DeepSeekAttentionLayerCoordinator::begin_decode(
    const DeepSeekRecentStateSubmission& recent,
    const DeepSeekCompressedLayerUpdateSubmission& update,
    const DeepSeekAttentionLayerSubmission& attention,
    DeepSeekRecentStateWriter& recent_writer,
    DeepSeekCompressedLayerUpdateCoordinator& update_coordinator,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (attention.kind == DeepSeekCompressedAttentionKind::kRecentOnly) {
    if (state_ != DeepSeekAttentionLayerCoordinatorState::kIdle ||
        recent.layer_id >= 2 ||
        (paged_selection_ && recent.layer_id != layer_id_) ||
        attention.query_positions.size() != 1 ||
        attention.query_positions[0] != recent.absolute_position ||
        attention.compressed_slot_count != 0 || attention.has_indexer_projection) {
      return Status::InvalidArgument("DeepSeek recent-only decode shape is invalid");
    }
    auto status = recent_writer.launch(recent, transaction);
    if (status.ok()) status = begin(attention, transaction);
    if (!status.ok()) state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return status;
  }
  if (state_ != DeepSeekAttentionLayerCoordinatorState::kIdle ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      (update.ratio != 4 && update.ratio != 128) ||
      recent.layer_id != update.main_state.layer_id ||
      recent.absolute_position != update.main_state.absolute_position ||
      recent.stream != transaction.stream() || recent.batch_count != 1 ||
      attention.query_positions.size() != 1 ||
      attention.query_positions[0] != update.main_state.absolute_position ||
      update.main_state.absolute_position >= 1048576 ||
      attention.attention.index_matrix != nullptr ||
      attention.attention.stream != transaction.stream()) {
    return Status::FailedPrecondition(
        "DeepSeek decode attention layer is not beginable");
  }
  const auto expected_kind =
      update.ratio == 4 ? DeepSeekCompressedAttentionKind::kRatio4
                        : DeepSeekCompressedAttentionKind::kRatio128;
  const auto expected_slots =
      (update.main_state.absolute_position + 1U) / update.ratio;
  if (attention.kind != expected_kind ||
      attention.compressed_slot_count != expected_slots) {
    return Status::InvalidArgument(
        "DeepSeek decode compressed coverage is inconsistent");
  }
  if (update.ratio == 4 && expected_slots != 0) {
    if (attention.selection.query_count != 1 ||
        attention.selection.total_slot_count != expected_slots ||
        attention.selection.visible_slot_counts.size() != 1 ||
        attention.selection.visible_slot_counts[0] != expected_slots ||
        attention.selection.stream != transaction.stream()) {
      return Status::InvalidArgument(
          "DeepSeek decode ratio-4 selection is inconsistent");
    }
  }
  auto status = recent_writer.launch(recent, transaction);
  if (!status.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return status;
  }
  status = update_coordinator.launch(update, transaction);
  if (!status.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return status;
  }
  status = begin(attention, transaction);
  if (!status.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
  }
  return status;
}

Status DeepSeekAttentionLayerCoordinator::launch_next_selection_tile() {
  if (state_ != DeepSeekAttentionLayerCoordinatorState::kSelectionReady) {
    return Status::FailedPrecondition(
        "DeepSeek attention index selection is not launchable");
  }
  auto status = selection_->launch_next_tile();
  if (!status.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return status;
  }
  state_ = DeepSeekAttentionLayerCoordinatorState::kSelectionInflight;
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus>
DeepSeekAttentionLayerCoordinator::poll_selection_tile() {
  if (state_ != DeepSeekAttentionLayerCoordinatorState::kSelectionInflight) {
    return Status::FailedPrecondition(
        "DeepSeek attention index selection is not inflight");
  }
  auto result = selection_->poll_tile();
  if (!result.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return result.status();
  }
  if (*result == DeepSeekExpertAsyncStatus::kInProgress) return *result;
  if (*result == DeepSeekExpertAsyncStatus::kError) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return *result;
  }
  if (*result != DeepSeekExpertAsyncStatus::kSuccess) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return Status::Internal(
        "DeepSeek attention selection returned invalid async state");
  }
  if (!selection_->complete()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kSelectionReady;
    return *result;
  }
  auto selected = selection_->finish();
  if (!selected.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return selected.status();
  }
  auto status = assemble_and_launch(*selected);
  if (!status.ok()) return status;
  return DeepSeekExpertAsyncStatus::kSuccess;
}

Status DeepSeekAttentionLayerCoordinator::reset() {
  if (state_ != DeepSeekAttentionLayerCoordinatorState::kAttentionPosted ||
      transaction_ == nullptr ||
      transaction_->state() !=
          DeepSeekAttentionSequenceTransactionState::kIdle) {
    return Status::FailedPrecondition(
        "DeepSeek attention layer coordinator is not resettable");
  }
  query_positions_.clear();
  submission_ = {};
  transaction_ = nullptr;
  state_ = DeepSeekAttentionLayerCoordinatorState::kIdle;
  return Status::Ok();
}

Status DeepSeekAttentionLayerCoordinator::assemble_and_launch(
    std::span<const std::vector<std::uint32_t>> selected) {
  Result<DeepSeekSparseIndexMatrix> matrix =
      submission_.kind == DeepSeekCompressedAttentionKind::kRecentOnly
          ? DeepSeekAttentionIndexAssembler::RecentOnly(
                query_positions_, submission_.recent_physical_offset)
          : submission_.kind == DeepSeekCompressedAttentionKind::kRatio4
          ? DeepSeekAttentionIndexAssembler::Ratio4(
                query_positions_, selected, submission_.compressed_slot_count,
                submission_.recent_physical_offset,
                submission_.compressed_physical_offset)
          : DeepSeekAttentionIndexAssembler::Ratio128(
                query_positions_, submission_.compressed_slot_count,
                submission_.recent_physical_offset,
                submission_.compressed_physical_offset);
  if (!matrix.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return matrix.status();
  }
  auto attention = submission_.attention;
  attention.index_matrix = &*matrix;
  Status status = Status::Ok();
  if (paged_selection_ && submission_.compressed_slot_count != 0) {
    auto bank = transaction_->tentative_fixed_state();
    if (!bank.ok()) {
      state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
      return bank.status();
    }
    auto layer = fixed_layout_.Resolve(layer_id_, *bank);
    if (!layer.ok()) {
      state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
      return layer.status();
    }
    const auto logical_pages =
        (submission_.compressed_slot_count + 63U) / 64U;
    Result<std::vector<std::uint32_t>> page_slots =
        submission_.kind == DeepSeekCompressedAttentionKind::kRatio4
            ? transaction_->ratio4_main_page_slots(layer_id_, logical_pages)
            : transaction_->ratio128_main_page_slots(layer_id_, logical_pages);
    if (!page_slots.ok()) {
      state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
      return page_slots.status();
    }
    attention.latent_kv_bf16 = layer->recent_bf16.address;
    attention.compressed_kv_bf16 =
        submission_.kind == DeepSeekCompressedAttentionKind::kRatio4
            ? page_arena_.ratio4_main_base()
            : page_arena_.ratio128_main_base();
    attention.recent_physical_offset = submission_.recent_physical_offset;
    attention.compressed_physical_offset =
        submission_.compressed_physical_offset;
    attention.compressed_slot_count = submission_.compressed_slot_count;
    const auto physical_pages =
        submission_.kind == DeepSeekCompressedAttentionKind::kRatio4
            ? transaction_->ratio4_physical_page_count()
            : transaction_->ratio128_physical_page_count();
    status = attention_->launch_paged(
        attention, *page_slots, physical_pages, *transaction_);
  } else {
    if (paged_selection_) {
      // Before the first compressed slot, decode still attends to the
      // committed recent history plus this token in the tentative bank.
      // The unpaged kernel indexes its KV pointer directly, so translate
      // logical recent indices to the zero-based 128-row ring here.
      auto bank = transaction_->tentative_fixed_state();
      if (!bank.ok()) {
        state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
        return bank.status();
      }
      auto layer = fixed_layout_.Resolve(layer_id_, *bank);
      if (!layer.ok()) {
        state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
        return layer.status();
      }
      for (auto& index : matrix->values) {
        if (index < 0) continue;
        const auto local = static_cast<std::int64_t>(index) -
                           submission_.recent_physical_offset;
        if (local < 0 || local >= 128) {
          state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
          return Status::InvalidArgument(
              "DeepSeek recent-only attention index is outside fixed state");
        }
        index = static_cast<std::int32_t>(local);
      }
      attention.latent_kv_bf16 = layer->recent_bf16.address;
      attention.kv_count = 128;
    }
    status = attention_->launch(attention, *transaction_);
  }
  if (!status.ok()) {
    state_ = DeepSeekAttentionLayerCoordinatorState::kPoisoned;
    return status;
  }
  state_ = DeepSeekAttentionLayerCoordinatorState::kAttentionPosted;
  return Status::Ok();
}

}  // namespace pih
