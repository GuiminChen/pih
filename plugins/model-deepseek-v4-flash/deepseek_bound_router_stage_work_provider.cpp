#include "pih/model/deepseek_bound_router_stage_work_provider.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& left,
                     const DeepSeekPipelinePlanDescriptor& right) {
  return left.engine_epoch == right.engine_epoch &&
         left.plan_sequence == right.plan_sequence && left.phase == right.phase &&
         left.token_count == right.token_count &&
         left.sequence_count == right.sequence_count;
}

}  // namespace

Result<DeepSeekBoundRouterStageWorkProvider>
DeepSeekBoundRouterStageWorkProvider::Create(
    DeepSeekStageRange owned_layers, DeepSeekExpertPlanStore& store) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42) {
    return Status::InvalidArgument(
        "DeepSeek bound router layer range is invalid");
  }
  DeepSeekBoundRouterStageWorkProvider result;
  result.owned_layers_ = owned_layers;
  result.store_ = &store;
  return result;
}

Status DeepSeekBoundRouterStageWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekBoundHashRouterWork> hash_work,
    std::span<const DeepSeekBoundLearnedRouterWork> learned_work) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek bound router plan identity is invalid");
  }
  std::array<std::optional<DeepSeekHashRouterStageWork>, 3> pending_hash;
  std::array<std::optional<DeepSeekLearnedRouterStageWork>, 43> pending_learned;
  std::uint32_t bound = 0;
  for (const auto& work : hash_work) {
    if (work.layer >= 3 || work.layer < owned_layers_.first_layer ||
        work.layer > owned_layers_.last_layer ||
        pending_hash[work.layer].has_value() ||
        work.token_ids.size() != descriptor.token_count ||
        ((work.coordinator == nullptr) != (work.scratch != nullptr)) ||
        (work.coordinator != nullptr &&
         (work.coordinator->plan_provider() != store_ ||
          work.submission.projection.layer != work.layer ||
          work.submission.projection.token_count != descriptor.token_count))) {
      return Status::InvalidArgument(
          "DeepSeek bound hash router work is invalid or duplicated");
    }
    pending_hash[work.layer] = DeepSeekHashRouterStageWork{
        work.token_ids, work.raw_scores, work.vocabulary_size,
        work.token_to_experts, work.scratch, store_, work.coordinator,
        work.submission};
    ++bound;
  }
  for (const auto& work : learned_work) {
    if (work.layer < 3 || work.layer < owned_layers_.first_layer ||
        work.layer > owned_layers_.last_layer ||
        pending_learned[work.layer].has_value() ||
        work.coordinator == nullptr ||
        work.coordinator->plan_provider() != store_ ||
        work.submission.layer != work.layer ||
        work.submission.token_count != descriptor.token_count) {
      return Status::InvalidArgument(
          "DeepSeek bound learned router work is invalid or duplicated");
    }
    pending_learned[work.layer] =
        DeepSeekLearnedRouterStageWork{work.coordinator, work.submission};
    ++bound;
  }
  const auto expected = owned_layers_.last_layer -
                        owned_layers_.first_layer + 1;
  if (bound != expected) {
    return Status::InvalidArgument(
        "DeepSeek bound router work does not cover every owned layer");
  }
  hash_ = std::move(pending_hash);
  learned_ = std::move(pending_learned);
  descriptor_ = descriptor;
  return Status::Ok();
}

void DeepSeekBoundRouterStageWorkProvider::clear() noexcept {
  descriptor_.reset();
  for (auto& work : hash_) work.reset();
  for (auto& work : learned_) work.reset();
}

Result<DeepSeekHashRouterStageWork>
DeepSeekBoundRouterStageWorkProvider::resolve_hash(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      command.kind != DeepSeekStageOperatorKind::kMoe || command.layer >= 3 ||
      !hash_[command.layer].has_value()) {
    return Status::FailedPrecondition(
        "DeepSeek hash router work is not bound to this command and plan");
  }
  return *hash_[command.layer];
}

Result<DeepSeekLearnedRouterStageWork>
DeepSeekBoundRouterStageWorkProvider::resolve_learned(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      command.kind != DeepSeekStageOperatorKind::kMoe || command.layer < 3 ||
      command.layer >= learned_.size() ||
      !learned_[command.layer].has_value()) {
    return Status::FailedPrecondition(
        "DeepSeek learned router work is not bound to this command and plan");
  }
  return *learned_[command.layer];
}

}  // namespace pih
