#include "pih/model/deepseek_bound_expert_plan_provider.h"

#include <array>

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

Result<DeepSeekBoundExpertPlanProvider>
DeepSeekBoundExpertPlanProvider::Create(DeepSeekStageRange owned_layers,
                                        std::uint32_t maximum_token_count) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_token_count == 0) {
    return Status::InvalidArgument(
        "DeepSeek bound expert provider capacity is invalid");
  }
  DeepSeekBoundExpertPlanProvider provider;
  provider.owned_layers_ = owned_layers;
  provider.maximum_token_count_ = maximum_token_count;
  const auto count = static_cast<std::size_t>(owned_layers.last_layer -
                                               owned_layers.first_layer + 1);
  for (auto& bank : provider.banks_) {
    bank.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto plan = DeepSeekExpertSubwavePlan::Reserve(maximum_token_count);
      if (!plan.ok()) return plan.status();
      bank.push_back(std::move(*plan));
    }
  }
  return provider;
}

Status DeepSeekBoundExpertPlanProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekLayerExpertRoutes> layers) {
  const auto layer_count = static_cast<std::size_t>(
      owned_layers_.last_layer - owned_layers_.first_layer + 1);
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 ||
      descriptor.token_count > maximum_token_count_ ||
      descriptor.sequence_count == 0 || layers.size() != layer_count ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek expert plan binding identity is invalid");
  }
  std::array<const DeepSeekLayerExpertRoutes*, 43> ordered{};
  for (const auto& layer : layers) {
    if (layer.layer < owned_layers_.first_layer ||
        layer.layer > owned_layers_.last_layer) {
      return Status::InvalidArgument(
          "DeepSeek expert plan binding contains a foreign layer");
    }
    const auto index = layer.layer - owned_layers_.first_layer;
    if (ordered[index] != nullptr) {
      return Status::InvalidArgument(
          "DeepSeek expert plan binding duplicates a layer");
    }
    ordered[index] = &layer;
  }
  const auto pending_bank = active_bank_ ^ 1U;
  for (std::size_t index = 0; index < layer_count; ++index) {
    const auto status = banks_[pending_bank][index].materialize(
        descriptor.token_count, ordered[index]->routes);
    if (!status.ok()) return status;
  }
  active_bank_ = pending_bank;
  active_descriptor_ = descriptor;
  published_.fill(false);
  for (std::uint32_t layer = owned_layers_.first_layer;
       layer <= owned_layers_.last_layer; ++layer) {
    published_[layer] = true;
  }
  return Status::Ok();
}

Status DeepSeekBoundExpertPlanProvider::begin(
    const DeepSeekPipelinePlanDescriptor& descriptor) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 ||
      descriptor.token_count > maximum_token_count_ ||
      descriptor.sequence_count == 0 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek expert plan store identity is invalid");
  }
  active_bank_ ^= 1U;
  active_descriptor_ = descriptor;
  published_.fill(false);
  return Status::Ok();
}

Status DeepSeekBoundExpertPlanProvider::publish_routes(
    std::uint32_t layer, std::uint32_t token_count,
    std::span<const DeepSeekExpertRoute> routes) {
  if (!active_descriptor_.has_value() ||
      layer < owned_layers_.first_layer || layer > owned_layers_.last_layer ||
      published_[layer] || token_count != active_descriptor_->token_count) {
    return Status::FailedPrecondition(
        "DeepSeek expert route publication is not valid for the active plan");
  }
  const auto index = layer - owned_layers_.first_layer;
  const auto status = banks_[active_bank_][index].materialize(
      token_count, routes);
  if (!status.ok()) return status;
  published_[layer] = true;
  return Status::Ok();
}

Result<const DeepSeekExpertSubwavePlan*>
DeepSeekBoundExpertPlanProvider::resolve(
    std::uint32_t layer,
    const DeepSeekPipelinePlanDescriptor& descriptor) {
  if (!active_descriptor_.has_value() ||
      !same_descriptor(*active_descriptor_, descriptor) ||
      layer < owned_layers_.first_layer || layer > owned_layers_.last_layer ||
      !published_[layer]) {
    return Status::FailedPrecondition(
        "DeepSeek expert work is not bound to this plan and layer");
  }
  return &banks_[active_bank_][layer - owned_layers_.first_layer];
}

}  // namespace pih
