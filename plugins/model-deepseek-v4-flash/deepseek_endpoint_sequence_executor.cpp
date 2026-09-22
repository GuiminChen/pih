#include "pih/model/deepseek_endpoint_sequence_executor.h"

namespace pih {

Result<DeepSeekEndpointSequenceExecutor>
DeepSeekEndpointSequenceExecutor::Create(
    DeepSeekEndpointSequenceOperations& operations,
    std::uint32_t* host_error_flag, std::uint32_t* host_sampled_token,
    float* host_selected_logprob, std::uint32_t* host_rng_word,
    std::uint32_t* host_top_ids, float* host_top_logprobs) {
  if (host_error_flag == nullptr || host_sampled_token == nullptr) {
    return Status::InvalidArgument("DeepSeek endpoint host error is null");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  status = operations.validate_host_error(host_sampled_token);
  if (!status.ok()) return status;
  if ((host_selected_logprob == nullptr) != (host_rng_word == nullptr)) {
    return Status::InvalidArgument(
        "DeepSeek sampling receipt host buffers are incomplete");
  }
  if ((host_top_ids == nullptr) != (host_top_logprobs == nullptr)) {
    return Status::InvalidArgument(
        "DeepSeek top-logprob host buffers are incomplete");
  }
  if (host_selected_logprob != nullptr) {
    status = operations.validate_host_error(
        reinterpret_cast<std::uint32_t*>(host_selected_logprob));
    if (!status.ok()) return status;
    status = operations.validate_host_error(host_rng_word);
    if (!status.ok()) return status;
  }
  if (host_top_ids != nullptr) {
    status = operations.validate_host_error(host_top_ids);
    if (!status.ok()) return status;
    status = operations.validate_host_error(
        reinterpret_cast<std::uint32_t*>(host_top_logprobs));
    if (!status.ok()) return status;
  }
  DeepSeekEndpointSequenceExecutor value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  value.host_sampled_token_ = host_sampled_token;
  value.host_selected_logprob_ = host_selected_logprob;
  value.host_rng_word_ = host_rng_word;
  value.host_top_ids_ = host_top_ids;
  value.host_top_logprobs_ = host_top_logprobs;
  return value;
}

Status DeepSeekEndpointSequenceExecutor::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}

Result<bool> DeepSeekEndpointSequenceExecutor::claim(
    std::uintptr_t device_error, std::uintptr_t stream,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != stream) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint sequence executor is not launchable");
  }
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, device_error);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) {
    *host_error_flag_ = 0;
    auto status = run(operations_->zero_u32_async(device_error, stream));
    if (!status.ok()) return status;
  }
  return *claimed;
}

Status DeepSeekEndpointSequenceExecutor::launch_embedding(
    DeepSeekEmbeddingLaunch launch,
    DeepSeekAttentionSequenceTransaction& transaction) {
  auto status = validate_deepseek_embedding_launch(launch);
  if (!status.ok()) return status;
  auto claimed = claim(launch.error_flag_u32, launch.stream, transaction);
  if (!claimed.ok()) return claimed.status();
  status = run(operations_->embedding(launch));
  if (!status.ok()) return status;
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, launch.error_flag_u32, launch.stream));
}

Status DeepSeekEndpointSequenceExecutor::launch_head(
    const DeepSeekHeadSequenceSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  auto status = validate_deepseek_hc_head_launch(submission.hc);
  if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(submission.rms);
  if (!status.ok()) return status;
  status = validate_deepseek_lm_head_launch(submission.lm);
  if (!status.ok()) return status;
  const bool stochastic = submission.stochastic_sample.has_value();
  if (stochastic) {
    status = validate_deepseek_stochastic_sample_launch(
        *submission.stochastic_sample);
  } else {
    status = validate_deepseek_argmax_launch(submission.sample);
  }
  if (!status.ok()) return status;
  const auto sample_error = stochastic
      ? submission.stochastic_sample->error_flag_u32
      : submission.sample.error_flag_u32;
  const auto sample_logits = stochastic
      ? submission.stochastic_sample->logits_f32
      : submission.sample.logits_f32;
  const auto sample_token = stochastic
      ? submission.stochastic_sample->token_id_u32
      : submission.sample.token_id_u32;
  const auto sample_stream = stochastic
      ? submission.stochastic_sample->stream
      : submission.sample.stream;
  const auto top_count = stochastic
      ? submission.stochastic_sample->top_logprobs_count
      : submission.sample.top_logprobs_count;
  const auto top_ids = stochastic
      ? submission.stochastic_sample->top_logprobs_ids_u32
      : submission.sample.top_logprobs_ids_u32;
  const auto top_values = stochastic
      ? submission.stochastic_sample->top_logprobs_f32
      : submission.sample.top_logprobs_f32;
  if (submission.hc.token_count != 1 || submission.rms.rows != 1 ||
      submission.lm.output_rows != 1 ||
      submission.hc.output_bf16 != submission.rms.input_bf16 ||
      submission.rms.output_bf16 != submission.lm.input_bf16 ||
      submission.hc.error_flag_u32 != submission.rms.error_flag ||
      submission.hc.error_flag_u32 != submission.lm.error_flag_u32 ||
      submission.hc.error_flag_u32 != sample_error ||
      submission.lm.logits_f32 != sample_logits ||
      submission.hc.stream != submission.rms.stream ||
      submission.hc.stream != submission.lm.stream ||
      submission.hc.stream != sample_stream ||
      (!stochastic && submission.sample.selected_logprob_f32 != 0 &&
       host_selected_logprob_ == nullptr) ||
      (top_count != 0 &&
       (host_top_ids_ == nullptr || host_top_logprobs_ == nullptr)) ||
      (stochastic &&
       (host_selected_logprob_ == nullptr || host_rng_word_ == nullptr))) {
    return Status::InvalidArgument(
        "DeepSeek head sequence dataflow is invalid");
  }
  auto claimed = claim(submission.hc.error_flag_u32, submission.hc.stream,
                       transaction);
  if (!claimed.ok()) return claimed.status();
  status = run(operations_->hc_head(submission.hc));
  if (!status.ok()) return status;
  status = run(operations_->rms_norm(submission.rms));
  if (!status.ok()) return status;
  status = run(operations_->lm_head(submission.lm));
  if (!status.ok()) return status;
  status = stochastic
      ? run(operations_->stochastic_sample(*submission.stochastic_sample))
      : run(operations_->sample(submission.sample));
  if (!status.ok()) return status;
  status = run(operations_->copy_token_d2h_async(
      host_sampled_token_, sample_token, sample_stream));
  if (!status.ok()) return status;
  if (stochastic || submission.sample.selected_logprob_f32 != 0) {
    status = run(operations_->copy_logprob_d2h_async(
        host_selected_logprob_,
        stochastic ? submission.stochastic_sample->selected_logprob_f32
                   : submission.sample.selected_logprob_f32,
        sample_stream));
    if (!status.ok()) return status;
  }
  if (top_count != 0) {
    status = run(operations_->copy_top_ids_d2h_async(
        host_top_ids_, top_ids, top_count, sample_stream));
    if (!status.ok()) return status;
    status = run(operations_->copy_top_logprobs_d2h_async(
        host_top_logprobs_, top_values, top_count, sample_stream));
    if (!status.ok()) return status;
  }
  if (stochastic) {
    status = run(operations_->copy_rng_d2h_async(
        host_rng_word_, submission.stochastic_sample->rng_word_u32,
        sample_stream));
    if (!status.ok()) return status;
  }
  return run(operations_->copy_error_d2h_async(
      host_error_flag_, submission.hc.error_flag_u32,
      submission.hc.stream));
}

}  // namespace pih
