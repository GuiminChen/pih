#include "pih/model/deepseek_endpoint_work_factory.h"

#include <unordered_set>

namespace pih {

Result<DeepSeekEndpointWorkFactory> DeepSeekEndpointWorkFactory::Create(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || maximum_sequences == 0) {
    return Status::InvalidArgument(
        "DeepSeek endpoint factory topology is invalid");
  }
  DeepSeekEndpointWorkFactory result;
  result.stage_ = stage;
  result.maximum_sequences_ = maximum_sequences;
  return result;
}

Status DeepSeekEndpointWorkFactory::append_plan_work(
    DeepSeekPlanPhase phase, std::uint32_t sequence_count,
    std::vector<DeepSeekEndpointStageSequenceWork> endpoint,
    DeepSeekRankComputeWorkBuilder& builder) const {
  if (sequence_count == 0 || sequence_count > maximum_sequences_ ||
      phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek endpoint plan identity is invalid");
  }
  const bool endpoint_required = stage_.owns_embedding || stage_.owns_lm_head;
  if (endpoint_required != !endpoint.empty() ||
      (endpoint_required && endpoint.size() != sequence_count)) {
    return Status::InvalidArgument(
        "DeepSeek endpoint work differs from stage ownership");
  }
  if (!endpoint_required) return Status::Ok();
  std::unordered_set<DeepSeekEndpointSequenceExecutor*> executors;
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> transactions;
  for (const auto& work : endpoint) {
    if (work.executor == nullptr || work.transaction == nullptr ||
        !executors.insert(work.executor).second ||
        !transactions.insert(work.transaction).second) {
      return Status::InvalidArgument(
          "DeepSeek endpoint packed state is incomplete or aliased");
    }
  }
  return builder.set_endpoint(std::move(endpoint));
}

}  // namespace pih
