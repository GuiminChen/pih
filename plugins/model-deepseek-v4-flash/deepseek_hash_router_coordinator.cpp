#include "pih/model/deepseek_hash_router_coordinator.h"

#include <limits>

namespace pih {

Result<DeepSeekHashRouterCoordinator> DeepSeekHashRouterCoordinator::Create(
    std::uint32_t maximum_tokens, DeepSeekRouteScratchArena& scratch,
    DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations) {
  if (maximum_tokens == 0 || maximum_tokens > 4096 ||
      scratch.maximum_token_count() < maximum_tokens) {
    return Status::InvalidArgument(
        "DeepSeek hash router projection capacity is invalid");
  }
  DeepSeekHashRouterCoordinator result;
  result.maximum_tokens_ = maximum_tokens;
  result.scratch_ = &scratch;
  result.store_ = &store;
  result.operations_ = &operations;
  return result;
}

Status DeepSeekHashRouterCoordinator::poison(Status status) {
  poisoned_ = true;
  inflight_ = false;
  return status.ok() ? Status::Internal("DeepSeek hash router poisoned")
                     : status;
}

Status DeepSeekHashRouterCoordinator::launch(
    const DeepSeekHashRouterSubmission& value) {
  const auto& projection = value.projection;
  const auto scores = static_cast<std::size_t>(projection.token_count) *
                      DeepSeekLearnedRouterCoordinator::kExpertCount;
  const auto table_rows = static_cast<std::size_t>(value.vocabulary_size) *
                          DeepSeekExpertSubwavePlan::kRoutesPerToken;
  if (inflight_ || poisoned_ || projection.layer >= 3 ||
      projection.token_count == 0 ||
      projection.token_count > maximum_tokens_ ||
      projection.input_bf16 == 0 || projection.weight_bf16 == 0 ||
      projection.scores_f32 == 0 || projection.error_flag_u32 == 0 ||
      projection.stream == 0 || projection.completion_event == 0 ||
      projection.host_scores.size() != scores ||
      projection.host_error_flag == nullptr ||
      value.token_ids.size() != projection.token_count ||
      value.vocabulary_size == 0 ||
      value.vocabulary_size >
          std::numeric_limits<std::size_t>::max() /
              DeepSeekExpertSubwavePlan::kRoutesPerToken ||
      value.token_to_experts.size() != table_rows) {
    return Status::FailedPrecondition(
        "DeepSeek hash router projection submission is invalid");
  }
  *projection.host_error_flag = 0;
  const auto run = [this](Status status) {
    return status.ok() ? status : poison(status);
  };
  auto status = run(operations_->zero_u32_async(
      projection.error_flag_u32, projection.stream));
  if (!status.ok()) return status;
  status = run(operations_->gemm(
      {projection.input_bf16, projection.weight_bf16,
       projection.scores_f32, projection.error_flag_u32,
       projection.stream, projection.token_count,
       DeepSeekLearnedRouterCoordinator::kExpertCount,
       DeepSeekLearnedRouterCoordinator::kHiddenSize}));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      projection.host_scores.data(), projection.scores_f32,
      scores * sizeof(float), projection.stream));
  if (!status.ok()) return status;
  status = run(operations_->copy_d2h_async(
      projection.host_error_flag, projection.error_flag_u32,
      sizeof(std::uint32_t), projection.stream));
  if (!status.ok()) return status;
  status = run(operations_->record_event(projection.completion_event,
                                          projection.stream));
  if (!status.ok()) return status;
  active_ = value;
  inflight_ = true;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus> DeepSeekHashRouterCoordinator::poll() {
  if (!inflight_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek hash router projection is not pollable");
  }
  const auto& projection = active_.projection;
  auto event = operations_->query_event(projection.completion_event);
  if (!event.ok()) return poison(event.status());
  if (*event == DeepSeekExpertAsyncStatus::kInProgress) {
    return DeepSeekStageComputeStatus::kInProgress;
  }
  if (*event == DeepSeekExpertAsyncStatus::kError ||
      *projection.host_error_flag != 0) {
    (void)poison(Status::Internal("DeepSeek hash router GPU failed"));
    return DeepSeekStageComputeStatus::kError;
  }
  if (*event != DeepSeekExpertAsyncStatus::kSuccess) {
    return poison(
        Status::Internal("DeepSeek hash router received invalid event state"));
  }
  auto status = DeepSeekHashRouter::RouteInto(
      projection.layer, active_.token_ids, projection.host_scores,
      active_.vocabulary_size, active_.token_to_experts, *scratch_, *store_);
  if (!status.ok()) return poison(status);
  inflight_ = false;
  return DeepSeekStageComputeStatus::kSuccess;
}

}  // namespace pih
