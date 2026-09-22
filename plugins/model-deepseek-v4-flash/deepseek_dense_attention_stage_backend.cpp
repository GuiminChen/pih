#include "pih/model/deepseek_dense_attention_stage_backend.h"

#include <unordered_set>

namespace pih { namespace {

Status validate_work(const DeepSeekDenseAttentionStageSequenceWork& work) {
  if (work.input_coordinator == nullptr || work.output_coordinator == nullptr ||
      work.transaction == nullptr || work.sparse_query_bf16 == 0 ||
      work.sparse_kv_bf16 == 0 || work.sparse_output_bf16 == 0) {
    return Status::InvalidArgument(
        "DeepSeek dense attention packed work is incomplete");
  }
  const auto tokens = work.input.input_quant.token_count;
  const auto stream = work.input.input_quant.stream;
  const auto error = work.input.input_quant.error_flag;
  if (tokens == 0 || work.input.q_rope.input_bf16 != work.sparse_query_bf16 ||
      work.input.kv_simulate.kv_bf16 != work.sparse_kv_bf16 ||
      work.output.inverse_rope.input_bf16 != work.sparse_output_bf16 ||
      work.output.inverse_rope.token_count != tokens ||
      work.output.inverse_rope.stream != stream ||
      work.output.inverse_rope.error_flag_u32 != error ||
      work.output.inverse_rope.positions_u32 !=
          work.input.q_rope.positions_u32 ||
      work.output.inverse_rope.table_position_count !=
          work.input.q_rope.table_position_count ||
      work.transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      work.transaction->stream() != stream) {
    return Status::InvalidArgument(
        "DeepSeek dense attention packed dataflow is inconsistent");
  }
  return Status::Ok();
}

}  // namespace pih::<anonymous>

Result<DeepSeekDenseAttentionStageOperatorBackend>
DeepSeekDenseAttentionStageOperatorBackend::Create(
    DeepSeekStageOperatorBackend& inner,
    DeepSeekDenseAttentionStageWorkProvider& provider) {
  DeepSeekDenseAttentionStageOperatorBackend value;
  value.inner_ = &inner;
  value.provider_ = &provider;
  return value;
}

Status DeepSeekDenseAttentionStageOperatorBackend::poison(Status status) {
  mode_ = Mode::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek dense attention stage poisoned")
                     : status;
}

Status DeepSeekDenseAttentionStageOperatorBackend::launch(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (mode_ != Mode::kIdle || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0) {
    return Status::FailedPrecondition(
        "DeepSeek dense attention stage backend is not launchable");
  }
  if (command.kind != DeepSeekStageOperatorKind::kAttention) {
    auto status = inner_->launch(command, plan);
    if (!status.ok()) return poison(status);
    mode_ = Mode::kPassthrough;
    return Status::Ok();
  }
  auto resolved = provider_->resolve(command, plan);
  if (!resolved.ok()) return poison(resolved.status());
  if (resolved->empty() || resolved->size() != plan.sequence_count) {
    return poison(Status::InvalidArgument(
        "DeepSeek dense attention work count does not match packed plan"));
  }
  std::unordered_set<DeepSeekAttentionProjectionCoordinator*> inputs;
  std::unordered_set<DeepSeekAttentionOutputProjectionCoordinator*> outputs;
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> transactions;
  for (const auto& item : *resolved) {
    auto status = validate_work(item);
    if (!status.ok() || !inputs.insert(item.input_coordinator).second ||
        !outputs.insert(item.output_coordinator).second ||
        !transactions.insert(item.transaction).second) {
      return poison(status.ok()
                        ? Status::InvalidArgument(
                              "DeepSeek dense attention work is duplicated")
                        : status);
    }
  }
  work_.assign(resolved->begin(), resolved->end());
  for (auto& item : work_) {
    auto status = item.input_coordinator->launch(item.input, *item.transaction);
    if (!status.ok()) return poison(status);
  }
  auto status = inner_->launch(command, plan);
  if (!status.ok()) return poison(status);
  mode_ = Mode::kAttention;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekDenseAttentionStageOperatorBackend::poll() {
  if (mode_ == Mode::kIdle || mode_ == Mode::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek dense attention stage backend is not pollable");
  }
  auto result = inner_->poll();
  if (!result.ok()) return poison(result.status());
  if (*result == DeepSeekStageComputeStatus::kInProgress) return *result;
  if (*result == DeepSeekStageComputeStatus::kError) {
    (void)poison(Status::Internal("DeepSeek sparse attention branch failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (mode_ == Mode::kAttention) {
    for (auto& item : work_) {
      auto status = item.output_coordinator->launch(item.output,
                                                    *item.transaction);
      if (!status.ok()) return poison(status);
    }
    work_.clear();
  }
  mode_ = Mode::kIdle;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
