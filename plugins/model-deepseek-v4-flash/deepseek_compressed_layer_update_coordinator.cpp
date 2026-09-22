#include "pih/model/deepseek_compressed_layer_update_coordinator.h"

#include <cmath>

#include "pih/model/deepseek_compressed_page_mutation_assembler.h"

namespace pih {
namespace {

bool same_state_epoch(const DeepSeekCompressorStateSubmission& left,
                      const DeepSeekCompressorStateSubmission& right) {
  return left.layer_id == right.layer_id &&
         left.device_error_flag_u32 == right.device_error_flag_u32 &&
         left.stream == right.stream &&
         left.batch_count == right.batch_count &&
         left.absolute_position == right.absolute_position;
}

bool valid_state(const DeepSeekCompressorStateSubmission& value,
                 const DeepSeekAttentionSequenceTransaction& transaction) {
  return value.kv_projection_f32 != 0 &&
         value.gate_projection_f32 != 0 && value.ape_row_f32 != 0 &&
         value.output_f32 != 0 && value.device_error_flag_u32 != 0 &&
         value.stream == transaction.stream() && value.batch_count == 1 &&
         value.absolute_position < 1048576;
}

Status validate_ratio4(
    const DeepSeekCompressedLayerUpdateSubmission& submission,
    const DeepSeekAttentionSequenceTransaction& transaction) {
  const auto& main = submission.main_state;
  const auto& index = submission.index_state;
  const bool boundary = (main.absolute_position + 1U) % 4U == 0;
  if (!valid_state(main, transaction) || !valid_state(index, transaction) ||
      main.indexer || !index.indexer || !same_state_epoch(main, index) ||
      boundary != submission.has_completed_slot) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 layer update is inconsistent");
  }
  if (!boundary) return Status::Ok();
  const auto& page = submission.ratio4_slot;
  if (page.main_compressed_f32 != main.output_f32 ||
      page.target.layer_id != main.layer_id ||
      (page.tail_cow && page.committed.layer_id != main.layer_id) ||
      page.index_compressed_f32 != index.output_f32 ||
      page.device_error_flag_u32 != main.device_error_flag_u32 ||
      page.stream != main.stream ||
      page.absolute_position != main.absolute_position) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 page update does not match compressor output");
  }
  auto owned = page.tail_cow
                   ? transaction.validate_ratio4_cow_target(page.committed,
                                                            page.target)
                   : transaction.validate_ratio4_append_target(page.target);
  if (!owned.ok()) return owned;
  auto status = validate_deepseek_compressor_bf16_store_launch(
      {page.main_compressed_f32, page.main_rms_weight_bf16,
       page.main_cos_sin_cache_f32, 1, page.device_error_flag_u32,
       page.stream, 512, 64, (page.absolute_position / 4U) * 4U,
       page.rms_epsilon});
  if (!status.ok()) return status;
  return validate_deepseek_compressor_bf16_store_launch(
      {page.index_compressed_f32, page.index_rms_weight_bf16,
       page.index_cos_sin_cache_f32, 1, page.device_error_flag_u32,
       page.stream, 128, 64, (page.absolute_position / 4U) * 4U,
       page.rms_epsilon});
}

Status validate_ratio128(
    const DeepSeekCompressedLayerUpdateSubmission& submission,
    const DeepSeekAttentionSequenceTransaction& transaction) {
  const auto& main = submission.main_state;
  const bool boundary = (main.absolute_position + 1U) % 128U == 0;
  if (!valid_state(main, transaction) || main.indexer ||
      boundary != submission.has_completed_slot) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 layer update is inconsistent");
  }
  if (!boundary) return Status::Ok();
  const auto& page = submission.ratio128_slot;
  if (page.compressed_f32 != main.output_f32 ||
      page.target.layer_id != main.layer_id ||
      (page.tail_cow && page.committed.layer_id != main.layer_id) ||
      page.device_error_flag_u32 != main.device_error_flag_u32 ||
      page.stream != main.stream ||
      page.absolute_position != main.absolute_position) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 page update does not match compressor output");
  }
  auto owned = page.tail_cow
                   ? transaction.validate_ratio128_cow_target(page.committed,
                                                              page.target)
                   : transaction.validate_ratio128_append_target(page.target);
  if (!owned.ok()) return owned;
  return validate_deepseek_compressor_bf16_store_launch(
      {page.compressed_f32, page.rms_weight_bf16,
       page.cos_sin_cache_f32, 1, page.device_error_flag_u32,
       page.stream, 512, 64, (page.absolute_position / 128U) * 128U,
       page.rms_epsilon});
}

Status validate_deferred(
    const DeepSeekCompressedLayerUpdateSubmission& submission,
    const DeepSeekAttentionSequenceTransaction& transaction) {
  const auto& resources = submission.deferred_page;
  if (submission.has_completed_slot ||
      !std::isfinite(resources.rms_epsilon) ||
      resources.rms_epsilon <= 0.0F ||
      !valid_state(submission.main_state, transaction) ||
      submission.main_state.indexer || resources.main_rms_weight_bf16 == 0 ||
      resources.main_cos_sin_cache_f32 == 0) {
    return Status::InvalidArgument(
        "DeepSeek deferred compressed page mutation is invalid");
  }
  if (submission.ratio == 4) {
    if (!valid_state(submission.index_state, transaction) ||
        !submission.index_state.indexer ||
        !same_state_epoch(submission.main_state, submission.index_state) ||
        resources.index_rms_weight_bf16 == 0 ||
        resources.index_cos_sin_cache_f32 == 0) {
      return Status::InvalidArgument(
          "DeepSeek deferred ratio-4 page mutation is invalid");
    }
    return Status::Ok();
  }
  if (submission.ratio != 128 ||
      submission.index_state.kv_projection_f32 != 0 ||
      resources.index_rms_weight_bf16 != 0 ||
      resources.index_cos_sin_cache_f32 != 0) {
    return Status::InvalidArgument(
        "DeepSeek deferred ratio-128 page mutation is invalid");
  }
  return Status::Ok();
}

Result<DeepSeekCompressedLayerUpdateSubmission> resolve_deferred(
    DeepSeekCompressedLayerUpdateSubmission submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (!submission.defer_page_mutation) return submission;
  auto status = validate_deferred(submission, transaction);
  if (!status.ok()) return status;
  const auto position = submission.main_state.absolute_position;
  const bool boundary = (position + 1U) % submission.ratio == 0;
  const auto stored_slot = position / submission.ratio;
  const bool tail = boundary && stored_slot % 64U != 0;
  const auto logical_page = stored_slot / 64U;
  std::optional<DeepSeekRatio4PagePair> committed4;
  std::optional<DeepSeekRatio128Page> committed128;
  if (tail && submission.ratio == 4) {
    auto found = transaction.find_published_ratio4(
        submission.main_state.layer_id, logical_page);
    if (!found.ok()) return found.status();
    if (!found->has_value())
      return Status::FailedPrecondition(
          "DeepSeek deferred ratio-4 tail is missing");
    committed4 = **found;
  } else if (tail) {
    auto found = transaction.find_published_ratio128(
        submission.main_state.layer_id, logical_page);
    if (!found.ok()) return found.status();
    if (!found->has_value())
      return Status::FailedPrecondition(
          "DeepSeek deferred ratio-128 tail is missing");
    committed128 = **found;
  }
  const auto& resources = submission.deferred_page;
  auto bound = DeepSeekCompressedPageMutationAssembler::BindDecode(
      submission, transaction, resources.main_rms_weight_bf16,
      resources.index_rms_weight_bf16, resources.main_cos_sin_cache_f32,
      resources.index_cos_sin_cache_f32, resources.rms_epsilon,
      committed4.has_value() ? &*committed4 : nullptr,
      committed128.has_value() ? &*committed128 : nullptr);
  if (!bound.ok()) return bound.status();
  bound->defer_page_mutation = false;
  return bound;
}

}  // namespace

Result<DeepSeekCompressedLayerUpdateCoordinator>
DeepSeekCompressedLayerUpdateCoordinator::Create(
    DeepSeekCompressorStateWriter& state_writer,
    DeepSeekCompressedPageWriter& page_writer) {
  DeepSeekCompressedLayerUpdateCoordinator coordinator;
  coordinator.state_writer_ = &state_writer;
  coordinator.page_writer_ = &page_writer;
  return coordinator;
}

Status DeepSeekCompressedLayerUpdateCoordinator::launch(
    const DeepSeekCompressedLayerUpdateSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  auto resolved = resolve_deferred(submission, transaction);
  if (!resolved.ok()) return resolved.status();
  auto status = validate(*resolved, transaction);
  if (!status.ok()) return status;
  auto run = [this](Status value) {
    if (!value.ok()) poisoned_ = true;
    return value;
  };
  status = run(state_writer_->launch(resolved->main_state, transaction));
  if (!status.ok()) return status;
  if (resolved->ratio == 4) {
    status = run(state_writer_->launch(resolved->index_state, transaction));
    if (!status.ok()) return status;
    return resolved->has_completed_slot
               ? run(page_writer_->write_ratio4(resolved->ratio4_slot,
                                                transaction))
               : Status::Ok();
  }
  return resolved->has_completed_slot
             ? run(page_writer_->write_ratio128(resolved->ratio128_slot,
                                                transaction))
             : Status::Ok();
}

Status DeepSeekCompressedLayerUpdateCoordinator::validate(
    const DeepSeekCompressedLayerUpdateSubmission& submission,
    const DeepSeekAttentionSequenceTransaction& transaction) const {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek compressed layer update is not launchable");
  }
  if (submission.defer_page_mutation) {
    return validate_deferred(submission, transaction);
  }
  return submission.ratio == 4
                    ? validate_ratio4(submission, transaction)
                    : submission.ratio == 128
                          ? validate_ratio128(submission, transaction)
                          : Status::InvalidArgument(
                                "DeepSeek compression ratio is invalid");
}

}  // namespace pih
