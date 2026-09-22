#include "pih/model/deepseek_bound_dspark_mtp_stage_work_provider.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& left,
                     const DeepSeekPipelinePlanDescriptor& right) {
  return left.engine_epoch == right.engine_epoch &&
         left.plan_sequence == right.plan_sequence &&
         left.phase == right.phase && left.token_count == right.token_count &&
         left.sequence_count == right.sequence_count;
}

bool valid_descriptor(const DeepSeekPipelinePlanDescriptor& descriptor) {
  return descriptor.engine_epoch != 0 && descriptor.plan_sequence != 0 &&
         descriptor.token_count != 0 && descriptor.sequence_count == 1 &&
         (descriptor.phase == DeepSeekPlanPhase::kPrefill ||
          (descriptor.phase == DeepSeekPlanPhase::kDecode &&
           descriptor.token_count == 1));
}

bool unresolved(const DeepSeekDsparkFixedStageDeviceView& state) {
  return state.recent_bf16.address == 0 && state.recent_bf16.bytes == 0;
}

bool valid_template(const DeepSeekBoundDsparkMtpStageWork& work,
                    DeepSeekDsparkStageId stage,
                    const DeepSeekPipelinePlanDescriptor& descriptor) {
  if (work.attention == nullptr) return false;
  if (descriptor.phase == DeepSeekPlanPhase::kPrefill) {
    const auto& value = work.prefill_resources;
    return work.moe == nullptr && value.stage == stage &&
           value.state_layout != nullptr && value.transaction != nullptr &&
           value.prepare_epoch == 0 && unresolved(value.recent_state) &&
           value.weights.wkv_fp8 != 0 &&
           value.weights.wkv_scale_ue8m0 != 0 &&
           value.weights.kv_norm_bf16 != 0 &&
           value.weights.generation != 0 &&
           value.weights.generation == value.weight_generation &&
           value.main_normalized_bf16 != 0 && value.activation_e4m3 != 0 &&
           value.activation_scale_ue8m0 != 0 && value.positions_u32 != 0 &&
           value.kv_scratch_bf16 != 0 &&
           value.rope_frequencies_f32 != 0 && value.error_flag_u32 != 0 &&
           value.stream != 0 && value.token_count == descriptor.token_count &&
           value.position_table_count >= value.token_count;
  }
  const auto& value = work.resources;
  return work.moe != nullptr && work.moe != work.attention &&
         value.stage == stage && value.weights != nullptr &&
         value.resident_experts != nullptr && value.state_layout != nullptr &&
         value.transaction != nullptr && value.prepare_epoch == 0 &&
         unresolved(value.recent_state) && value.weight_generation != 0;
}

}  // namespace

Result<DeepSeekBoundDsparkMtpStageWorkProvider>
DeepSeekBoundDsparkMtpStageWorkProvider::Create(bool owns_dspark) {
  if (!owns_dspark) {
    return Status::InvalidArgument(
        "DeepSeek bound DSpark MTP provider requires DSpark ownership");
  }
  return DeepSeekBoundDsparkMtpStageWorkProvider{};
}

Status DeepSeekBoundDsparkMtpStageWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekBoundDsparkMtpStageWork> stages) {
  if (!valid_descriptor(descriptor) || stages.size() != 3) {
    return Status::InvalidArgument(
        "DeepSeek bound DSpark MTP plan shape is invalid");
  }
  const auto pending = active_bank_ ^ 1U;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  const DeepSeekFixedStateLayout* layout = nullptr;
  for (std::size_t index = 0; index < stages.size(); ++index) {
    const auto stage = static_cast<DeepSeekDsparkStageId>(index);
    if (!valid_template(stages[index], stage, descriptor)) {
      return Status::InvalidArgument(
          "DeepSeek bound DSpark MTP stage template is invalid");
    }
    auto* candidate_transaction = descriptor.phase == DeepSeekPlanPhase::kPrefill
        ? stages[index].prefill_resources.transaction
        : stages[index].resources.transaction;
    auto* candidate_layout = descriptor.phase == DeepSeekPlanPhase::kPrefill
        ? stages[index].prefill_resources.state_layout
        : stages[index].resources.state_layout;
    if ((transaction != nullptr && candidate_transaction != transaction) ||
        (layout != nullptr && candidate_layout != layout)) {
      return Status::InvalidArgument(
          "DeepSeek bound DSpark MTP stages disagree on transaction state");
    }
    transaction = candidate_transaction;
    layout = candidate_layout;
    templates_[pending][index] = stages[index];
  }
  resolved_[pending] = templates_[pending];
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<const DeepSeekBoundDsparkMtpStageWork*>
DeepSeekBoundDsparkMtpStageWorkProvider::resolve(
    DeepSeekDsparkStageId stage,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      !is_valid_deepseek_dspark_stage(stage)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MTP work is not bound to this plan");
  }
  const auto index = static_cast<std::size_t>(stage);
  auto& work = resolved_[active_bank_][index];
  work = templates_[active_bank_][index];
  auto* transaction = plan.phase == DeepSeekPlanPhase::kPrefill
      ? work.prefill_resources.transaction
      : work.resources.transaction;
  const auto* layout = plan.phase == DeepSeekPlanPhase::kPrefill
      ? work.prefill_resources.state_layout
      : work.resources.state_layout;
  if (transaction == nullptr || layout == nullptr ||
      transaction->state() !=
          DeepSeekAttentionSequenceTransactionState::kPreparing ||
      transaction->prepare_epoch() == 0) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark MTP tentative state is not preparing");
  }
  auto bank = transaction->tentative_fixed_state();
  if (!bank.ok()) return bank.status();
  auto recent = layout->ResolveDspark(stage, *bank);
  if (!recent.ok()) return recent.status();
  if (plan.phase == DeepSeekPlanPhase::kPrefill) {
    if (work.prefill_resources.stream != transaction->stream()) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark MTP stream differs from transaction");
    }
    work.prefill_resources.prepare_epoch = transaction->prepare_epoch();
    work.prefill_resources.recent_state = *recent;
  } else {
    work.resources.prepare_epoch = transaction->prepare_epoch();
    work.resources.recent_state = *recent;
  }
  return &work;
}

}  // namespace pih
