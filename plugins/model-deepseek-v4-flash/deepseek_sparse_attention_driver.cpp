#include "pih/model/deepseek_sparse_attention_driver.h"

#include <algorithm>

namespace pih {

Result<DeepSeekSparseAttentionDriver> DeepSeekSparseAttentionDriver::Create(
    DeepSeekSparseAttentionOperations& operations,
    DeepSeekSparseAttentionHostStaging staging) {
  if (staging.indices == nullptr || staging.error_flag == nullptr ||
      staging.index_capacity == 0 ||
      (staging.page_slots == nullptr) != (staging.page_slot_capacity == 0)) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention staging is invalid");
  }
  auto status = operations.validate_host_staging(staging);
  if (!status.ok()) return status;
  DeepSeekSparseAttentionDriver driver;
  driver.operations_ = &operations;
  driver.staging_ = staging;
  return driver;
}

Status DeepSeekSparseAttentionDriver::launch(
    const DeepSeekSparseAttentionSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  return launch_impl(submission, {}, 0, transaction);
}

Status DeepSeekSparseAttentionDriver::launch_paged(
    const DeepSeekSparseAttentionSubmission& submission,
    std::span<const std::uint32_t> page_slots,
    std::uint32_t physical_page_count,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (page_slots.empty() || page_slots.size() > staging_.page_slot_capacity ||
      physical_page_count == 0 || submission.compressed_kv_bf16 == 0 ||
      submission.device_page_slots_u32 == 0 ||
      submission.compressed_slot_count == 0 ||
      page_slots.size() != (submission.compressed_slot_count + 63U) / 64U) {
    return Status::InvalidArgument(
        "DeepSeek paged sparse attention submission is invalid");
  }
  for (const auto slot : page_slots) {
    if (slot >= physical_page_count) {
      return Status::InvalidArgument(
          "DeepSeek paged sparse attention page slot is invalid");
    }
  }
  return launch_impl(submission, page_slots, physical_page_count, transaction);
}

Status DeepSeekSparseAttentionDriver::launch_impl(
    const DeepSeekSparseAttentionSubmission& submission,
    std::span<const std::uint32_t> page_slots,
    std::uint32_t physical_page_count,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (submission.index_matrix == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention index matrix is missing");
  }
  auto status = validate(submission, *submission.index_matrix, transaction);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      staging_.error_flag, submission.device_error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *staging_.error_flag = 0;
  }
  std::copy(submission.index_matrix->values.begin(),
            submission.index_matrix->values.end(), staging_.indices);
  std::copy(page_slots.begin(), page_slots.end(), staging_.page_slots);
  auto run = [this](Status value) {
    if (!value.ok()) poisoned_ = true;
    return value;
  };
  status = Status::Ok();
  if (*claimed) {
    status = run(operations_->zero_u32_async(
        submission.device_error_flag_u32, submission.stream));
    if (!status.ok()) return status;
  }
  status = run(operations_->copy_h2d_async(
      submission.device_indices_i32, staging_.indices,
      submission.index_matrix->values.size() * sizeof(std::int32_t),
      submission.stream));
  if (!status.ok()) return status;
  if (!page_slots.empty()) {
    status = run(operations_->copy_h2d_async(
        submission.device_page_slots_u32, staging_.page_slots,
        page_slots.size_bytes(), submission.stream));
    if (!status.ok()) return status;
  }
  DeepSeekSparseAttentionLaunch launch{
      submission.query_bf16, submission.latent_kv_bf16,
      submission.attention_sink_f32, submission.device_indices_i32,
      submission.output_bf16, submission.device_error_flag_u32,
      submission.stream, submission.index_matrix->query_count,
      submission.head_count, submission.kv_count,
      submission.index_matrix->row_width};
  if (!page_slots.empty()) {
    launch.compressed_kv_bf16 = submission.compressed_kv_bf16;
    launch.page_slots_u32 = submission.device_page_slots_u32;
    launch.recent_physical_offset = submission.recent_physical_offset;
    launch.compressed_physical_offset = submission.compressed_physical_offset;
    launch.compressed_slot_count = submission.compressed_slot_count;
    launch.logical_page_count = static_cast<std::uint32_t>(page_slots.size());
    launch.physical_page_count = physical_page_count;
  }
  status = run(operations_->attention(
      launch));
  if (!status.ok()) return status;
  return run(operations_->copy_d2h_async(
      staging_.error_flag, submission.device_error_flag_u32,
      sizeof(std::uint32_t), submission.stream));
}

Status DeepSeekSparseAttentionDriver::validate(
    const DeepSeekSparseAttentionSubmission& submission,
    const DeepSeekSparseIndexMatrix& index_matrix,
    const DeepSeekAttentionSequenceTransaction& transaction) const {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek sparse attention driver is not launchable");
  }
  if ((submission.index_matrix != nullptr &&
       submission.index_matrix != &index_matrix) ||
      submission.query_bf16 == 0 ||
      submission.latent_kv_bf16 == 0 ||
      submission.attention_sink_f32 == 0 ||
      submission.device_indices_i32 == 0 || submission.output_bf16 == 0 ||
      submission.device_error_flag_u32 == 0 || submission.stream == 0 ||
      submission.stream != transaction.stream() ||
      submission.head_count != 64 || submission.kv_count == 0 ||
      index_matrix.query_count == 0 || index_matrix.row_width == 0 ||
      index_matrix.row_width > 8320 ||
      index_matrix.values.size() !=
          static_cast<std::size_t>(index_matrix.query_count) *
              index_matrix.row_width ||
      index_matrix.values.size() > staging_.index_capacity) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention submission is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
