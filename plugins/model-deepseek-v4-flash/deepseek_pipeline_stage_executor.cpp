#include "pih/model/deepseek_pipeline_stage_executor.h"

namespace pih {
namespace {

Status validate_executor_shape(
    const DeepSeekPipelineTransaction& transaction,
    DeepSeekPipelineTransactionState expected_state,
    const DeepSeekStagePlan& stage,
    std::uint32_t world_size
    ) {
  if (transaction.state() != expected_state || world_size < 1 ||
      world_size > 4 || stage.rank >= world_size ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 ||
      world_size != 1 ||
      stage.owns_embedding != (stage.rank == 0) ||
      stage.owns_lm_head != (stage.rank + 1 == world_size) ||
      (stage.owns_dspark && stage.rank + 1 != world_size)) {
    return Status::InvalidArgument("DeepSeek pipeline stage topology is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekPipelineStageExecutor> DeepSeekPipelineStageExecutor::Create(
    const DeepSeekPipelineTransaction& transaction, DeepSeekStagePlan stage,
    std::uint32_t world_size
    ) {
  const auto status = validate_executor_shape(
      transaction, DeepSeekPipelineTransactionState::kCommitted, stage,
      world_size
      );
  if (!status.ok()) return status;
  return DeepSeekPipelineStageExecutor(
      transaction, stage, world_size
      );
}

Result<DeepSeekPipelineStageExecutor>
DeepSeekPipelineStageExecutor::CreateStaged(
    const DeepSeekPipelineTransaction& transaction, DeepSeekStagePlan stage,
    std::uint32_t world_size
    ) {
  const auto status = validate_executor_shape(
      transaction, DeepSeekPipelineTransactionState::kPrepared, stage,
      world_size
      );
  if (!status.ok()) return status;
  return DeepSeekPipelineStageExecutor(
      transaction, stage, world_size
      );
}

Status DeepSeekPipelineStageExecutor::poison(Status status) {
  state_ = DeepSeekPipelineStageExecutorState::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek pipeline stage poisoned")
                     : status;
}


Status DeepSeekPipelineStageExecutor::launch_compute(
    DeepSeekStageComputeDriver& driver) {
  const auto status = driver.launch(transaction_->descriptor(), stage_);
  if (!status.ok()) return poison(status);
  state_ = DeepSeekPipelineStageExecutorState::kCompute;
  return Status::Ok();
}

Status DeepSeekPipelineStageExecutor::advance(
    DeepSeekStageExecutionDrivers& drivers) {
  if (state_ == DeepSeekPipelineStageExecutorState::kComplete) return Status::Ok();
  if (state_ == DeepSeekPipelineStageExecutorState::kPoisoned) {
    return Status::Internal("DeepSeek pipeline stage is poisoned");
  }
  if (transaction_->state() != DeepSeekPipelineTransactionState::kCommitted ||
      drivers.compute == nullptr) {
    return poison(Status::FailedPrecondition(
        "DeepSeek stage lost committed transaction or compute driver"));
  }
  if (state_ == DeepSeekPipelineStageExecutorState::kReady) {
    return launch_compute(*drivers.compute);
  }
  if (state_ == DeepSeekPipelineStageExecutorState::kCompute) {
    auto result = drivers.compute->poll();
    if (!result.ok()) return poison(result.status());
    if (*result == DeepSeekStageComputeStatus::kError) {
      return poison(Status::Internal("DeepSeek stage compute failed"));
    }
    if (*result == DeepSeekStageComputeStatus::kInProgress) {
      return Status::Unavailable("DeepSeek stage compute remains in progress");
    }
    state_ = DeepSeekPipelineStageExecutorState::kComplete;
    return Status::Ok();
  }
  return poison(Status::Internal("DeepSeek PP1 stage entered an invalid state"));
}

}  // namespace pih
