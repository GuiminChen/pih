#include "pih/model/deepseek_compressor_state_writer.h"

#include <utility>

namespace pih {

Result<DeepSeekCompressorStateWriter> DeepSeekCompressorStateWriter::Create(
    DeepSeekFixedStateLayout layout,
    DeepSeekCompressorStateOperations& operations,
    std::uint32_t* host_error_flag) {
  if (layout.total_bytes() == 0 || host_error_flag == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek compressor state writer resources are invalid");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekCompressorStateWriter writer;
  writer.layout_ = std::move(layout);
  writer.operations_ = &operations;
  writer.host_error_flag_ = host_error_flag;
  return writer;
}

Status DeepSeekCompressorStateWriter::launch(
    const DeepSeekCompressorStateSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      submission.stream != transaction.stream() ||
      submission.kv_projection_f32 == 0 ||
      submission.gate_projection_f32 == 0 || submission.ape_row_f32 == 0 ||
      submission.output_f32 == 0 || submission.device_error_flag_u32 == 0 ||
      submission.batch_count != 1 || submission.absolute_position >= 1048576) {
    return Status::FailedPrecondition(
        "DeepSeek compressor state writer is not launchable");
  }
  auto bank = transaction.tentative_fixed_state();
  if (!bank.ok()) return bank.status();
  auto layer = layout_.Resolve(submission.layer_id, *bank);
  if (!layer.ok()) return layer.status();

  std::uint32_t ratio = 0;
  std::uint32_t head_dim = 0;
  DeepSeekExpertArenaSpan kv_state;
  DeepSeekExpertArenaSpan score_state;
  if (layer->kind == DeepSeekFixedLayerKind::kRatio4) {
    ratio = 4;
    head_dim = submission.indexer ? 128U : 512U;
    kv_state = submission.indexer ? layer->index_kv_state_f32
                                  : layer->main_kv_state_f32;
    score_state = submission.indexer ? layer->index_score_state_f32
                                     : layer->main_score_state_f32;
  } else if (layer->kind == DeepSeekFixedLayerKind::kRatio128 &&
             !submission.indexer) {
    ratio = 128;
    head_dim = 512;
    kv_state = layer->main_kv_state_f32;
    score_state = layer->main_score_state_f32;
  } else {
    return Status::InvalidArgument(
        "DeepSeek layer has no requested compressor state");
  }
  if (kv_state.address == 0 || score_state.address == 0) {
    return Status::Internal("DeepSeek compressor state view is empty");
  }
  const bool projected = submission.projection.input_bf16 != 0 ||
                         submission.projection.kv_weight_bf16 != 0 ||
                         submission.projection.gate_weight_bf16 != 0;
  if (projected) {
    const auto& projection = submission.projection;
    if (projection.input_bf16 == 0 || projection.kv_weight_bf16 == 0 || projection.gate_weight_bf16 == 0 ||
        projection.kv_projection_f32 != submission.kv_projection_f32 ||
        projection.gate_projection_f32 != submission.gate_projection_f32 ||
        projection.error_flag_u32 != submission.device_error_flag_u32 ||
        projection.stream != submission.stream ||
        projection.token_count != submission.batch_count ||
        projection.ratio != ratio || projection.head_dim != head_dim ||
        projection.hidden_size != 4096) {
      return Status::InvalidArgument(
          "DeepSeek compressor projection does not match state submission");
    }
    auto projection_status =
        validate_deepseek_compressor_bf16_projection_launch(projection);
    if (!projection_status.ok()) return projection_status;
  }
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, submission.device_error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) *host_error_flag_ = 0;
  auto run = [this](Status status) {
    if (!status.ok()) poisoned_ = true;
    return status;
  };
  if (*claimed) {
    auto status = run(operations_->zero_u32_async(
        submission.device_error_flag_u32, submission.stream));
    if (!status.ok()) return status;
  }
  if (projected) {
    auto status = run(operations_->projection(submission.projection));
    if (!status.ok()) return status;
  }
  auto status = run(operations_->pooling(
      {submission.kv_projection_f32, submission.gate_projection_f32,
       submission.ape_row_f32, kv_state.address, score_state.address,
       submission.output_f32, submission.device_error_flag_u32,
       submission.stream, submission.batch_count, ratio, head_dim,
       submission.absolute_position}));
  if (!status.ok()) return status;
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, submission.device_error_flag_u32,
      submission.stream));
}

}  // namespace pih
