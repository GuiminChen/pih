#include "pih/model/deepseek_compressed_page_writer.h"

#include <algorithm>
#include <utility>

namespace pih {
namespace {

constexpr std::uint64_t kSlotsPerPage = 64;

Result<std::uint32_t> slot_in_page(std::uint32_t absolute_position,
                                   std::uint32_t ratio,
                                   std::uint32_t logical_page) {
  const auto consumed = static_cast<std::uint64_t>(absolute_position) + 1U;
  if (consumed % ratio != 0) {
    return Status::FailedPrecondition(
        "DeepSeek compressed slot boundary is incomplete");
  }
  const auto ordinal = consumed / ratio - 1U;
  if (ordinal / kSlotsPerPage != logical_page) {
    return Status::FailedPrecondition(
        "DeepSeek compressed slot logical page is inconsistent");
  }
  return static_cast<std::uint32_t>(ordinal % kSlotsPerPage);
}

DeepSeekCompressorBf16StoreLaunch store_launch(
    std::uintptr_t compressed, std::uintptr_t weight,
    std::uintptr_t cos_sin, std::uintptr_t destination,
    std::uintptr_t error, std::uintptr_t stream, std::uint32_t head_dim,
    std::uint32_t position, std::uint32_t ratio, float epsilon) {
  return {compressed, weight, cos_sin, destination, error, stream,
          head_dim, 64, (position / ratio) * ratio, epsilon};
}

}  // namespace

Result<DeepSeekCompressedPageWriter> DeepSeekCompressedPageWriter::Create(
    DeepSeekAttentionPageArena arena,
    DeepSeekCompressedPageOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek compressed page host error is null");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekCompressedPageWriter writer;
  writer.arena_ = std::move(arena);
  writer.operations_ = &operations;
  writer.host_error_flag_ = host_error_flag;
  return writer;
}

Status DeepSeekCompressedPageWriter::write_ratio4(
    const DeepSeekRatio4SlotSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || submission.stream != transaction.stream()) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 page writer is not launchable");
  }
  auto owned = submission.tail_cow
                   ? transaction.validate_ratio4_cow_target(
                         submission.committed, submission.target)
                   : transaction.validate_ratio4_append_target(
                         submission.target);
  if (!owned.ok()) return owned;
  auto slot = slot_in_page(submission.absolute_position, 4,
                           submission.target.logical_page);
  if (!slot.ok()) return slot.status();
  auto main_page = arena_.Resolve(submission.target.main);
  auto index_page = arena_.Resolve(submission.target.index);
  if (!main_page.ok()) return main_page.status();
  if (!index_page.ok()) return index_page.status();
  const auto main_destination = main_page->address + *slot * 512U * 2U;
  const auto index_destination = index_page->address + *slot * 128U * 2U;
  auto main = store_launch(
      submission.main_compressed_f32, submission.main_rms_weight_bf16,
      submission.main_cos_sin_cache_f32, main_destination,
      submission.device_error_flag_u32, submission.stream, 512,
      submission.absolute_position, 4, submission.rms_epsilon);
  auto index = store_launch(
      submission.index_compressed_f32, submission.index_rms_weight_bf16,
      submission.index_cos_sin_cache_f32, index_destination,
      submission.device_error_flag_u32, submission.stream, 128,
      submission.absolute_position, 4, submission.rms_epsilon);
  auto status = validate_deepseek_compressor_bf16_store_launch(main);
  if (!status.ok()) return status;
  status = validate_deepseek_compressor_bf16_store_launch(index);
  if (!status.ok()) return status;
  DeepSeekExpertArenaSpan old_main;
  DeepSeekExpertArenaSpan old_index;
  if (copied_epoch_ != transaction.prepare_epoch()) {
    copied_epoch_ = transaction.prepare_epoch();
    copied_tail_targets_.clear();
  }
  const bool copy_tail = submission.tail_cow &&
      std::find(copied_tail_targets_.begin(), copied_tail_targets_.end(),
                submission.target.main.value) == copied_tail_targets_.end();
  if (copy_tail) {
    auto value = arena_.Resolve(submission.committed.main);
    if (!value.ok()) return value.status();
    old_main = *value;
    value = arena_.Resolve(submission.committed.index);
    if (!value.ok()) return value.status();
    old_index = *value;
  }
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, submission.device_error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  auto run = [this](Status value) {
    if (!value.ok()) poisoned_ = true;
    return value;
  };
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        submission.device_error_flag_u32, submission.stream));
    if (!status.ok()) return status;
  }
  if (copy_tail) {
    status = run(operations_->copy_d2d_async(
        main_page->address, old_main.address,
        static_cast<std::size_t>(main_page->bytes), submission.stream));
    if (!status.ok()) return status;
    status = run(operations_->copy_d2d_async(
        index_page->address, old_index.address,
        static_cast<std::size_t>(index_page->bytes), submission.stream));
    if (!status.ok()) return status;
    copied_tail_targets_.push_back(submission.target.main.value);
  }
  status = run(operations_->store(main));
  if (!status.ok()) return status;
  status = run(operations_->store(index));
  if (!status.ok()) return status;
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, submission.device_error_flag_u32,
      submission.stream));
}

Status DeepSeekCompressedPageWriter::write_ratio128(
    const DeepSeekRatio128SlotSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || submission.stream != transaction.stream()) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 page writer is not launchable");
  }
  auto owned = submission.tail_cow
                   ? transaction.validate_ratio128_cow_target(
                         submission.committed, submission.target)
                   : transaction.validate_ratio128_append_target(
                         submission.target);
  if (!owned.ok()) return owned;
  auto slot = slot_in_page(submission.absolute_position, 128,
                           submission.target.logical_page);
  if (!slot.ok()) return slot.status();
  auto page = arena_.Resolve(submission.target.handle);
  if (!page.ok()) return page.status();
  const auto destination = page->address + *slot * 512U * 2U;
  auto launch = store_launch(
      submission.compressed_f32, submission.rms_weight_bf16,
      submission.cos_sin_cache_f32, destination,
      submission.device_error_flag_u32, submission.stream, 512,
      submission.absolute_position, 128, submission.rms_epsilon);
  auto status = validate_deepseek_compressor_bf16_store_launch(launch);
  if (!status.ok()) return status;
  DeepSeekExpertArenaSpan old_page;
  if (copied_epoch_ != transaction.prepare_epoch()) {
    copied_epoch_ = transaction.prepare_epoch();
    copied_tail_targets_.clear();
  }
  const bool copy_tail = submission.tail_cow &&
      std::find(copied_tail_targets_.begin(), copied_tail_targets_.end(),
                submission.target.handle.value) == copied_tail_targets_.end();
  if (copy_tail) {
    auto value = arena_.Resolve(submission.committed.handle);
    if (!value.ok()) return value.status();
    old_page = *value;
  }
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, submission.device_error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  auto run = [this](Status value) {
    if (!value.ok()) poisoned_ = true;
    return value;
  };
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        submission.device_error_flag_u32, submission.stream));
    if (!status.ok()) return status;
  }
  if (copy_tail) {
    status = run(operations_->copy_d2d_async(
        page->address, old_page.address,
        static_cast<std::size_t>(page->bytes), submission.stream));
    if (!status.ok()) return status;
    copied_tail_targets_.push_back(submission.target.handle.value);
  }
  status = run(operations_->store(launch));
  if (!status.ok()) return status;
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, submission.device_error_flag_u32,
      submission.stream));
}

}  // namespace pih
