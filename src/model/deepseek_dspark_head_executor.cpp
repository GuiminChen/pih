#include "pih/model/deepseek_dspark_head_executor.h"

namespace pih { namespace {

Status validate_flow(const DeepSeekDsparkHeadSubmission& s) {
  auto status = validate_deepseek_hc_head_launch(s.hc);
  if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(s.rms);
  if (!status.ok()) return status;
  status = validate_deepseek_lm_head_launch(s.lm);
  if (!status.ok()) return status;
  const auto error = s.markov[0].error_flag_u32;
  const auto stream = s.markov[0].stream;
  const auto embedding_weight = s.markov[0].embedding_weight_bf16;
  const auto head_weight = s.markov[0].head_weight_bf16;
  const auto raw_base = s.markov[0].raw_logits_f32;
  const auto biased_base = s.markov[0].biased_logits_f32;
  const auto embed_base = s.markov[0].markov_embeddings_bf16;
  constexpr std::uintptr_t kLogitStride = 129280ULL * sizeof(float);
  constexpr std::uintptr_t kEmbedStride = 256ULL * 2ULL;
  if (s.hc.token_count != s.kBlockSize || s.rms.rows != s.kBlockSize ||
      s.lm.output_rows != s.kBlockSize ||
      s.hc.output_bf16 != s.rms.input_bf16 ||
      s.rms.output_bf16 != s.lm.input_bf16 ||
      s.lm.logits_f32 != raw_base ||
      s.hc.error_flag_u32 != error || s.rms.error_flag != error ||
      s.lm.error_flag_u32 != error || s.hc.stream != stream ||
      s.rms.stream != stream || s.lm.stream != stream) {
    return Status::InvalidArgument(
        "DeepSeek DSpark projection head dataflow is invalid");
  }
  for (std::size_t index = 0; index < s.kBlockSize; ++index) {
    status = validate_deepseek_dspark_markov_launch(s.markov[index]);
    if (!status.ok()) return status;
    status = validate_deepseek_argmax_launch(s.argmax[index]);
    if (!status.ok()) return status;
    const auto& markov = s.markov[index];
    const auto& argmax = s.argmax[index];
    if (markov.row_count != 1 || markov.error_flag_u32 != error ||
        markov.stream != stream || argmax.error_flag_u32 != error ||
        argmax.stream != stream ||
        markov.embedding_weight_bf16 != embedding_weight ||
        markov.head_weight_bf16 != head_weight ||
        markov.raw_logits_f32 != raw_base + index * kLogitStride ||
        markov.biased_logits_f32 != biased_base + index * kLogitStride ||
        markov.markov_embeddings_bf16 != embed_base + index * kEmbedStride ||
        argmax.logits_f32 != markov.biased_logits_f32 ||
        (index + 1 < s.kBlockSize &&
         argmax.token_id_u32 != s.markov[index + 1].token_ids_u32)) {
      return Status::InvalidArgument(
          "DeepSeek DSpark causal head dataflow is invalid");
    }
  }
  status = validate_deepseek_dspark_confidence_launch(s.confidence);
  if (!status.ok()) return status;
  if (s.confidence.row_count != s.kBlockSize ||
      s.confidence.markov_embeddings_bf16 != embed_base ||
      s.confidence.error_flag_u32 != error || s.confidence.stream != stream) {
    return Status::InvalidArgument(
        "DeepSeek DSpark confidence dataflow is invalid");
  }
  return Status::Ok();
}

}  // namespace pih::<anonymous>

Result<DeepSeekDsparkHeadExecutor> DeepSeekDsparkHeadExecutor::Create(
    DeepSeekDsparkHeadOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr)
    return Status::InvalidArgument("DeepSeek DSpark host error is null");
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekDsparkHeadExecutor value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}

Status DeepSeekDsparkHeadExecutor::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}

Status DeepSeekDsparkHeadExecutor::launch(
    const DeepSeekDsparkHeadSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != submission.markov[0].stream) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark head executor is not launchable");
  }
  auto status = validate_flow(submission);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, submission.markov[0].error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *host_error_flag_ = 0;
    status = run(operations_->zero_u32_async(
        submission.markov[0].error_flag_u32,
        submission.markov[0].stream));
    if (!status.ok()) return status;
  }
  status = run(operations_->hc_head(submission.hc));
  if (!status.ok()) return status;
  status = run(operations_->rms_norm(submission.rms));
  if (!status.ok()) return status;
  status = run(operations_->lm_head(submission.lm));
  if (!status.ok()) return status;
  for (std::size_t index = 0; index < submission.kBlockSize; ++index) {
    status = run(operations_->markov(submission.markov[index]));
    if (!status.ok()) return status;
    status = run(operations_->argmax(submission.argmax[index]));
    if (!status.ok()) return status;
  }
  status = run(operations_->confidence(submission.confidence));
  if (!status.ok()) return status;
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, submission.markov[0].error_flag_u32,
      submission.markov[0].stream));
}

}  // namespace pih
