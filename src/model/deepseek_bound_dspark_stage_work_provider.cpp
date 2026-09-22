#include "pih/model/deepseek_bound_dspark_stage_work_provider.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& a,
                     const DeepSeekPipelinePlanDescriptor& b) {
  return a.engine_epoch == b.engine_epoch && a.plan_sequence == b.plan_sequence &&
         a.phase == b.phase && a.token_count == b.token_count &&
         a.sequence_count == b.sequence_count;
}

}  // namespace

Result<DeepSeekBoundDsparkStageWorkProvider>
DeepSeekBoundDsparkStageWorkProvider::Create(
    bool owns_dspark, std::uint32_t maximum_sequences) {
  if (!owns_dspark || maximum_sequences == 0) {
    return Status::InvalidArgument(
        "DeepSeek bound DSpark provider capacity is invalid");
  }
  DeepSeekBoundDsparkStageWorkProvider provider;
  provider.maximum_sequences_ = maximum_sequences;
  return provider;
}

Status DeepSeekBoundDsparkStageWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    DeepSeekDsparkStageWork work) {
  const bool phase = descriptor.phase == DeepSeekPlanPhase::kPrefill ||
                     descriptor.phase == DeepSeekPlanPhase::kDecode;
  const auto expected_kind = descriptor.phase == DeepSeekPlanPhase::kPrefill
      ? DeepSeekDsparkStageWorkKind::kPrefillStateInitialization
      : DeepSeekDsparkStageWorkKind::kDecodeProposal;
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 ||
      descriptor.sequence_count > maximum_sequences_ || !phase ||
      work.kind != expected_kind || work.embed_coordinator == nullptr ||
      work.transaction == nullptr ||
      (descriptor.phase == DeepSeekPlanPhase::kDecode &&
       work.head_executor == nullptr)) {
    return Status::InvalidArgument(
        "DeepSeek bound DSpark plan work is invalid");
  }
  const auto pending = active_bank_ ^ 1U;
  banks_[pending] = std::move(work);
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<const DeepSeekDsparkStageWork*>
DeepSeekBoundDsparkStageWorkProvider::resolve(
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark work is not bound to this plan");
  }
  return &banks_[active_bank_];
}

}  // namespace pih
