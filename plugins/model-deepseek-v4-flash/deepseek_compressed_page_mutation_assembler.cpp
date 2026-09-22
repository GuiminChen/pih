#include "pih/model/deepseek_compressed_page_mutation_assembler.h"

#include <cmath>

namespace pih { namespace {

bool valid_state(const DeepSeekCompressorStateSubmission& state,
                 const DeepSeekAttentionSequenceTransaction& transaction) {
  return state.layer_id <= 42 && state.kv_projection_f32 != 0 &&
         state.gate_projection_f32 != 0 && state.ape_row_f32 != 0 &&
         state.output_f32 != 0 && state.device_error_flag_u32 != 0 &&
         state.stream == transaction.stream() && state.batch_count == 1 &&
         state.absolute_position < 1048576;
}

bool same_epoch(const DeepSeekCompressorStateSubmission& main,
                const DeepSeekCompressorStateSubmission& index) {
  return main.layer_id == index.layer_id &&
         main.absolute_position == index.absolute_position &&
         main.device_error_flag_u32 == index.device_error_flag_u32 &&
         main.stream == index.stream && main.batch_count == index.batch_count;
}

}  // namespace

Result<DeepSeekCompressedLayerUpdateSubmission>
DeepSeekCompressedPageMutationAssembler::BindDecode(
    DeepSeekCompressedLayerUpdateSubmission update,
    DeepSeekAttentionSequenceTransaction& transaction,
    std::uintptr_t main_rms_weight_bf16,
    std::uintptr_t index_rms_weight_bf16,
    std::uintptr_t main_cos_sin_cache_f32,
    std::uintptr_t index_cos_sin_cache_f32,
    float rms_epsilon,
    const DeepSeekRatio4PagePair* committed_ratio4,
    const DeepSeekRatio128Page* committed_ratio128) {
  if (transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      !valid_state(update.main_state, transaction) ||
      update.main_state.indexer || update.has_completed_slot ||
      !std::isfinite(rms_epsilon) || rms_epsilon <= 0.0F) {
    return Status::InvalidArgument(
        "DeepSeek compressed page mutation input is invalid");
  }
  const auto position = update.main_state.absolute_position;
  if (update.ratio == 4) {
    if (!valid_state(update.index_state, transaction) ||
        !update.index_state.indexer ||
        !same_epoch(update.main_state, update.index_state)) {
      return Status::InvalidArgument(
          "DeepSeek ratio-4 compressed page state pair is invalid");
    }
    if ((position + 1U) % 4U != 0) {
      if (committed_ratio4 != nullptr || committed_ratio128 != nullptr)
        return Status::InvalidArgument(
            "DeepSeek remainder update cannot bind a compressed page");
      return update;
    }
    if (main_rms_weight_bf16 == 0 || index_rms_weight_bf16 == 0 ||
        main_cos_sin_cache_f32 == 0 || index_cos_sin_cache_f32 == 0 ||
        committed_ratio128 != nullptr) {
      return Status::InvalidArgument(
          "DeepSeek ratio-4 page store resources are incomplete");
    }
    const auto stored_slot = position / 4U;
    const auto logical_page = stored_slot / 64U;
    const bool tail = stored_slot % 64U != 0;
    if (tail != (committed_ratio4 != nullptr)) {
      return Status::InvalidArgument(
          "DeepSeek ratio-4 tail COW source is inconsistent");
    }
    Result<DeepSeekRatio4PagePair> target = tail
        ? transaction.reserve_ratio4_tail_cow(*committed_ratio4)
        : transaction.reserve_ratio4_append(update.main_state.layer_id,
                                             logical_page);
    if (!target.ok()) return target.status();
    update.has_completed_slot = true;
    update.ratio4_slot = {
        *target, tail ? *committed_ratio4 : DeepSeekRatio4PagePair{}, tail,
        update.main_state.output_f32, update.index_state.output_f32,
        main_rms_weight_bf16, index_rms_weight_bf16,
        main_cos_sin_cache_f32, index_cos_sin_cache_f32,
        update.main_state.device_error_flag_u32, update.main_state.stream,
        position, rms_epsilon};
    return update;
  }
  if (update.ratio != 128 || update.index_state.kv_projection_f32 != 0) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 compressed page state is invalid");
  }
  if ((position + 1U) % 128U != 0) {
    if (committed_ratio4 != nullptr || committed_ratio128 != nullptr)
      return Status::InvalidArgument(
          "DeepSeek remainder update cannot bind a compressed page");
    return update;
  }
  if (main_rms_weight_bf16 == 0 || main_cos_sin_cache_f32 == 0 ||
      index_rms_weight_bf16 != 0 || index_cos_sin_cache_f32 != 0 ||
      committed_ratio4 != nullptr) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 page store resources are invalid");
  }
  const auto stored_slot = position / 128U;
  const auto logical_page = stored_slot / 64U;
  const bool tail = stored_slot % 64U != 0;
  if (tail != (committed_ratio128 != nullptr)) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 tail COW source is inconsistent");
  }
  Result<DeepSeekRatio128Page> target = tail
      ? transaction.reserve_ratio128_tail_cow(*committed_ratio128)
      : transaction.reserve_ratio128_append(update.main_state.layer_id,
                                             logical_page);
  if (!target.ok()) return target.status();
  update.has_completed_slot = true;
  update.ratio128_slot = {
      *target, tail ? *committed_ratio128 : DeepSeekRatio128Page{}, tail,
      update.main_state.output_f32, main_rms_weight_bf16,
      main_cos_sin_cache_f32, update.main_state.device_error_flag_u32,
      update.main_state.stream, position, rms_epsilon};
  return update;
}

Result<std::vector<DeepSeekCompressedLayerUpdateSubmission>>
DeepSeekCompressedPageMutationAssembler::BindChunk(
    std::vector<DeepSeekCompressedLayerUpdateSubmission> updates,
    DeepSeekAttentionSequenceTransaction& transaction,
    std::uintptr_t main_rms_weight_bf16,
    std::uintptr_t index_rms_weight_bf16,
    std::uintptr_t main_cos_sin_cache_f32,
    std::uintptr_t index_cos_sin_cache_f32,
    float rms_epsilon) {
  if (transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      updates.empty() || updates.size() > 4096 ||
      !std::isfinite(rms_epsilon) || rms_epsilon <= 0.0F) {
    return Status::InvalidArgument(
        "DeepSeek compressed page mutation chunk is invalid");
  }
  const auto ratio = updates.front().ratio;
  const auto layer = updates.front().main_state.layer_id;
  const auto first_position = updates.front().main_state.absolute_position;
  const auto error = updates.front().main_state.device_error_flag_u32;
  const auto stream = updates.front().main_state.stream;
  if ((ratio != 4 && ratio != 128) ||
      main_rms_weight_bf16 == 0 || main_cos_sin_cache_f32 == 0 ||
      (ratio == 4 && (index_rms_weight_bf16 == 0 ||
                      index_cos_sin_cache_f32 == 0)) ||
      (ratio == 128 && (index_rms_weight_bf16 != 0 ||
                        index_cos_sin_cache_f32 != 0))) {
    return Status::InvalidArgument(
        "DeepSeek compressed page mutation chunk resources are invalid");
  }
  for (std::size_t token = 0; token < updates.size(); ++token) {
    const auto& update = updates[token];
    const auto expected_position =
        static_cast<std::uint64_t>(first_position) + token;
    if (expected_position >= 1048576 || update.ratio != ratio ||
        update.has_completed_slot || update.defer_page_mutation ||
        !valid_state(update.main_state, transaction) ||
        update.main_state.indexer ||
        update.main_state.layer_id != layer ||
        update.main_state.absolute_position != expected_position ||
        update.main_state.device_error_flag_u32 != error ||
        update.main_state.stream != stream ||
        (ratio == 4 && (!valid_state(update.index_state, transaction) ||
                        !update.index_state.indexer ||
                        !same_epoch(update.main_state, update.index_state))) ||
        (ratio == 128 && update.index_state.kv_projection_f32 != 0)) {
      return Status::InvalidArgument(
          "DeepSeek compressed page mutation chunk token is invalid");
    }
  }

  std::optional<std::uint32_t> active_logical_page;
  DeepSeekRatio4SlotSubmission active_ratio4;
  DeepSeekRatio128SlotSubmission active_ratio128;
  auto rollback = [&transaction, &active_logical_page](Status failure) {
    if (!active_logical_page.has_value()) return failure;
    auto aborted = transaction.cancel_preparing();
    return aborted.ok() ? failure : aborted;
  };
  for (auto& update : updates) {
    const auto position = update.main_state.absolute_position;
    if ((position + 1U) % ratio != 0) continue;
    const auto stored_slot = position / ratio;
    const auto logical_page = stored_slot / 64U;
    if (!active_logical_page.has_value() ||
        *active_logical_page != logical_page) {
      const bool tail = stored_slot % 64U != 0;
      std::optional<DeepSeekRatio4PagePair> committed4;
      std::optional<DeepSeekRatio128Page> committed128;
      if (tail && ratio == 4) {
        auto found = transaction.find_published_ratio4(layer, logical_page);
        if (!found.ok()) return rollback(found.status());
        if (!found->has_value()) {
          return rollback(Status::FailedPrecondition(
              "DeepSeek ratio-4 chunk tail is missing"));
        }
        committed4 = **found;
      } else if (tail) {
        auto found = transaction.find_published_ratio128(layer, logical_page);
        if (!found.ok()) return rollback(found.status());
        if (!found->has_value()) {
          return rollback(Status::FailedPrecondition(
              "DeepSeek ratio-128 chunk tail is missing"));
        }
        committed128 = **found;
      }
      auto bound = BindDecode(
          update, transaction, main_rms_weight_bf16,
          index_rms_weight_bf16, main_cos_sin_cache_f32,
          index_cos_sin_cache_f32, rms_epsilon,
          committed4.has_value() ? &*committed4 : nullptr,
          committed128.has_value() ? &*committed128 : nullptr);
      if (!bound.ok()) return rollback(bound.status());
      update = std::move(*bound);
      active_logical_page = logical_page;
      if (ratio == 4) active_ratio4 = update.ratio4_slot;
      else active_ratio128 = update.ratio128_slot;
      continue;
    }
    update.has_completed_slot = true;
    if (ratio == 4) {
      update.ratio4_slot = active_ratio4;
      update.ratio4_slot.main_compressed_f32 = update.main_state.output_f32;
      update.ratio4_slot.index_compressed_f32 = update.index_state.output_f32;
      update.ratio4_slot.absolute_position = position;
    } else {
      update.ratio128_slot = active_ratio128;
      update.ratio128_slot.compressed_f32 = update.main_state.output_f32;
      update.ratio128_slot.absolute_position = position;
    }
  }
  return updates;
}

}  // namespace pih
