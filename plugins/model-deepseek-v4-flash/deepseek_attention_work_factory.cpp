#include "pih/model/deepseek_attention_work_factory.h"

namespace pih {
namespace {

template <typename Input, typename Append>
Status append_exact(
    DeepSeekStageRange owned,
    const std::array<
        std::optional<DeepSeekAttentionLayerRuntimeResources>, 43>& resources,
    std::vector<Input> layers, Append append) {
  const auto expected = owned.last_layer - owned.first_layer + 1;
  if (layers.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek attention plan does not cover every owned layer");
  }
  std::array<bool, 43> seen{};
  for (auto& layer : layers) {
    if (layer.layer < owned.first_layer || layer.layer > owned.last_layer ||
        seen[layer.layer] || !resources[layer.layer].has_value() ||
        layer.transaction == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek attention plan layer is foreign or duplicated");
    }
    seen[layer.layer] = true;
    const auto status = append(*resources[layer.layer], std::move(layer));
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekAttentionWorkFactory>
DeepSeekAttentionWorkFactory::Create(
    DeepSeekStageRange owned_layers,
    std::vector<DeepSeekAttentionLayerRuntimeResources> layers) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 ||
      layers.size() != owned_layers.last_layer - owned_layers.first_layer + 1) {
    return Status::InvalidArgument(
        "DeepSeek attention runtime layer coverage is invalid");
  }
  DeepSeekAttentionWorkFactory result;
  result.owned_layers_ = owned_layers;
  std::array<bool, 43> seen{};
  for (auto& layer : layers) {
    if (layer.layer < owned_layers.first_layer ||
        layer.layer > owned_layers.last_layer || seen[layer.layer] ||
        layer.recent_writer == nullptr ||
        layer.update_coordinator == nullptr ||
        layer.attention_coordinator == nullptr ||
        layer.prefill_coordinator == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek attention runtime resource is incomplete or duplicated");
    }
    seen[layer.layer] = true;
    result.layers_[layer.layer] = layer;
  }
  return result;
}

Status DeepSeekAttentionWorkFactory::append_decode_plan_work(
    std::vector<DeepSeekDecodeAttentionLayerPlanInput> layers,
    DeepSeekRankComputeWorkBuilder& builder) const {
  return append_exact(
      owned_layers_, layers_, std::move(layers),
      [&builder](const auto& resource, auto input) {
        DeepSeekDecodeAttentionWork work;
        work.recent = input.recent;
        work.update = input.update;
        work.attention = input.attention;
        work.recent_writer = resource.recent_writer;
        work.update_coordinator = resource.update_coordinator;
        work.coordinator = resource.attention_coordinator;
        work.transaction = input.transaction;
        return builder.add_decode_attention(input.layer, std::move(work));
      });
}

Status DeepSeekAttentionWorkFactory::append_chunk_plan_work(
    std::vector<DeepSeekChunkAttentionLayerPlanInput> layers,
    DeepSeekRankComputeWorkBuilder& builder) const {
  return append_exact(
      owned_layers_, layers_, std::move(layers),
      [&builder](const auto& resource, auto input) {
        DeepSeekChunkAttentionWork work;
        work.submission = input.submission;
        work.chunk_coordinator = resource.prefill_coordinator;
        work.attention_coordinator = resource.attention_coordinator;
        work.transaction = input.transaction;
        return builder.add_chunk_attention(input.layer, std::move(work));
      });
}

}  // namespace pih
