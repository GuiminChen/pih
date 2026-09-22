#include "pih/model/deepseek_bound_endpoint_stage_work_provider.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& left,
                     const DeepSeekPipelinePlanDescriptor& right) {
  return left.engine_epoch == right.engine_epoch &&
         left.plan_sequence == right.plan_sequence && left.phase == right.phase &&
         left.token_count == right.token_count &&
         left.sequence_count == right.sequence_count;
}

Status validate_work(
    std::span<const DeepSeekEndpointStageSequenceWork> work) {
  for (std::size_t index = 0; index < work.size(); ++index) {
    if (work[index].executor == nullptr || work[index].transaction == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek bound endpoint work is incomplete");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (work[index].executor == work[prior].executor ||
          work[index].transaction == work[prior].transaction) {
        return Status::InvalidArgument(
            "DeepSeek bound endpoint work aliases packed sequence state");
      }
    }
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekBoundEndpointStageWorkProvider>
DeepSeekBoundEndpointStageWorkProvider::Create(
    std::uint32_t maximum_sequences, bool owns_embedding,
    bool owns_lm_head) {
  if (maximum_sequences == 0 || (!owns_embedding && !owns_lm_head)) {
    return Status::InvalidArgument(
        "DeepSeek bound endpoint provider capacity is invalid");
  }
  DeepSeekBoundEndpointStageWorkProvider provider;
  provider.maximum_sequences_ = maximum_sequences;
  provider.owns_embedding_ = owns_embedding;
  provider.owns_lm_head_ = owns_lm_head;
  for (auto& bank : provider.banks_) bank.reserve(maximum_sequences);
  return provider;
}

Status DeepSeekBoundEndpointStageWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekEndpointStageSequenceWork> work) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.sequence_count == 0 ||
      descriptor.sequence_count > maximum_sequences_ ||
      work.size() != descriptor.sequence_count ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek bound endpoint plan identity is invalid");
  }
  const auto status = validate_work(work);
  if (!status.ok()) return status;
  const auto pending = active_bank_ ^ 1U;
  banks_[pending].assign(work.begin(), work.end());
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<std::span<const DeepSeekEndpointStageSequenceWork>>
DeepSeekBoundEndpointStageWorkProvider::resolve(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  const bool allowed =
      (command.kind == DeepSeekStageOperatorKind::kEmbedding &&
       owns_embedding_) ||
      (command.kind == DeepSeekStageOperatorKind::kHead && owns_lm_head_);
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      !allowed || command.layer != 0) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint work is not bound to this command and plan");
  }
  return std::span<const DeepSeekEndpointStageSequenceWork>(
      banks_[active_bank_]);
}

}  // namespace pih
