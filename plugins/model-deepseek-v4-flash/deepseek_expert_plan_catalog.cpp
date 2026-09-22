#include "pih/model/deepseek_expert_plan_catalog.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& left,
                     const DeepSeekPipelinePlanDescriptor& right) {
  return left.engine_epoch == right.engine_epoch &&
         left.plan_sequence == right.plan_sequence &&
         left.phase == right.phase && left.token_count == right.token_count &&
         left.sequence_count == right.sequence_count;
}

}  // namespace

Result<DeepSeekExpertPlanCatalog> DeepSeekExpertPlanCatalog::Create(
    DeepSeekPipelinePlanDescriptor descriptor,
    DeepSeekStageRange owned_layers) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain ||
      owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43) {
    return Status::InvalidArgument(
        "DeepSeek expert plan catalog identity is invalid");
  }
  DeepSeekExpertPlanCatalog catalog;
  catalog.descriptor_ = descriptor;
  catalog.owned_layers_ = owned_layers;
  for (std::uint32_t layer = owned_layers.first_layer;
       layer <= owned_layers.last_layer; ++layer) {
    auto plan = DeepSeekExpertSubwavePlan::Reserve(descriptor.token_count);
    if (!plan.ok()) return plan.status();
    catalog.plans_[layer] = std::move(*plan);
  }
  return catalog;
}

Status DeepSeekExpertPlanCatalog::publish(
    std::uint32_t layer, DeepSeekExpertSubwavePlan plan) {
  return publish_routes(layer, plan.token_count(), plan.routes());
}

Status DeepSeekExpertPlanCatalog::publish_routes(
    std::uint32_t layer, std::uint32_t token_count,
    std::span<const DeepSeekExpertRoute> routes) {
  if (layer < owned_layers_.first_layer || layer > owned_layers_.last_layer ||
      token_count != descriptor_.token_count || published_[layer]) {
    return Status::FailedPrecondition(
        "DeepSeek expert plan publication is invalid");
  }
  const auto status = plans_[layer]->materialize(token_count, routes);
  if (!status.ok()) return status;
  published_[layer] = true;
  ++published_layers_;
  return Status::Ok();
}

Result<const DeepSeekExpertSubwavePlan*> DeepSeekExpertPlanCatalog::resolve(
    std::uint32_t layer,
    const DeepSeekPipelinePlanDescriptor& descriptor) {
  if (!same_descriptor(descriptor_, descriptor) ||
      layer < owned_layers_.first_layer || layer > owned_layers_.last_layer ||
      !plans_[layer].has_value() || !published_[layer]) {
    return Status::FailedPrecondition(
        "DeepSeek expert plan is absent or stale");
  }
  return &*plans_[layer];
}

}  // namespace pih
