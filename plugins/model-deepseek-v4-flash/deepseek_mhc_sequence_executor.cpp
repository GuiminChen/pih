#include "pih/model/deepseek_mhc_sequence_executor.h"

namespace pih {

Result<DeepSeekMhcSequenceExecutor> DeepSeekMhcSequenceExecutor::Create(
    DeepSeekMhcSequenceOperations& operations,
    std::uint32_t* host_error_flag) {
  if (host_error_flag == nullptr) {
    return Status::InvalidArgument("DeepSeek mHC host error flag is null");
  }
  auto status = operations.validate_host_error(host_error_flag);
  if (!status.ok()) return status;
  DeepSeekMhcSequenceExecutor value;
  value.operations_ = &operations;
  value.host_error_flag_ = host_error_flag;
  return value;
}

Status DeepSeekMhcSequenceExecutor::launch(
    const DeepSeekMhcSequenceSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  auto status = begin(submission, transaction);
  if (!status.ok()) return status;
  status = operations_->branch(active_branch_);
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  return finish(transaction);
}

Status DeepSeekMhcSequenceExecutor::begin(
    const DeepSeekMhcSequenceSubmission& submission,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || active_transaction_ != nullptr ||
      transaction.state() !=
                       DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction.stream() != submission.stream) {
    return Status::FailedPrecondition(
        "DeepSeek mHC sequence executor is not launchable");
  }
  const DeepSeekMhcPreLaunch pre{
      submission.residual_bf16,
      submission.fn_f32,
      submission.scale_f32,
      submission.base_f32,
      submission.norm_weight_bf16,
      submission.post_mix_f32,
      submission.residual_mix_f32,
      submission.layer_input_bf16,
      submission.device_error_flag_u32,
      submission.stream,
      submission.token_count,
      4096,
      submission.rms_epsilon,
      submission.pre_epsilon,
      submission.sinkhorn_epsilon,
      2.0F,
      submission.sinkhorn_iterations};
  const DeepSeekMhcBranchLaunch branch{
      submission.kind, submission.layer_id, submission.layer_input_bf16,
      submission.branch_output_bf16, submission.device_error_flag_u32,
      submission.stream, submission.token_count, 4096};
  const DeepSeekMhcPostLaunch post{
      submission.branch_output_bf16, submission.residual_bf16,
      submission.post_mix_f32, submission.residual_mix_f32,
      submission.output_bf16, submission.device_error_flag_u32,
      submission.stream, submission.token_count, 4096};
  DeepSeekMhcTargetHiddenTapLaunch target_hidden_tap;
  if (submission.target_hidden_bf16 != 0) {
    target_hidden_tap = {
        submission.output_bf16, submission.target_hidden_bf16,
        submission.device_error_flag_u32, submission.stream,
        submission.token_count, 4096, 4, 3,
        submission.target_stage_index};
  }
  auto status = validate_deepseek_mhc_sequence_submission(submission);
  if (!status.ok()) return status;
  auto claimed = transaction.claim_external_error_channel(
      host_error_flag_, submission.device_error_flag_u32);
  if (!claimed.ok()) return claimed.status();
  if (*claimed) *host_error_flag_ = 0;
  auto run = [this](Status result) {
    if (!result.ok()) poisoned_ = true;
    return result;
  };
  if (*claimed) {
    status = run(operations_->zero_u32_async(
        submission.device_error_flag_u32, submission.stream));
    if (!status.ok()) return status;
  }
  status = run(operations_->pre(pre));
  if (!status.ok()) return status;
  active_transaction_ = &transaction;
  active_branch_ = branch;
  active_post_ = post;
  active_target_hidden_tap_ = target_hidden_tap;
  return Status::Ok();
}

Status DeepSeekMhcSequenceExecutor::finish(
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (poisoned_ || active_transaction_ != &transaction ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing) {
    return Status::FailedPrecondition(
        "DeepSeek mHC sequence executor has no completed branch");
  }
  auto run = [this](Status result) {
    if (!result.ok()) poisoned_ = true;
    return result;
  };
  auto status = run(operations_->post(active_post_));
  if (!status.ok()) return status;
  if (active_target_hidden_tap_.target_hidden_bf16 != 0) {
    status = run(operations_->target_hidden_tap(active_target_hidden_tap_));
    if (!status.ok()) return status;
  }
  status = run(operations_->copy_error_d2h_async(
      host_error_flag_, active_post_.error_flag_u32, active_post_.stream));
  if (!status.ok()) return status;
  active_transaction_ = nullptr;
  active_branch_ = {};
  active_post_ = {};
  active_target_hidden_tap_ = {};
  return Status::Ok();
}

}  // namespace pih
