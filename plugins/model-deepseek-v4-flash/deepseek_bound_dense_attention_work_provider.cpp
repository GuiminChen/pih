#include "pih/model/deepseek_bound_dense_attention_work_provider.h"

#include <limits>

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& a,
                     const DeepSeekPipelinePlanDescriptor& b) {
  return a.engine_epoch == b.engine_epoch && a.plan_sequence == b.plan_sequence &&
         a.phase == b.phase && a.token_count == b.token_count &&
         a.sequence_count == b.sequence_count;
}

Status validate_layer(
    const DeepSeekBoundDenseAttentionLayerWork& layer,
    const DeepSeekPipelinePlanDescriptor& descriptor) {
  if (layer.sequences.size() != descriptor.sequence_count) {
    return Status::InvalidArgument(
        "DeepSeek dense attention packed sequence count is invalid");
  }
  std::uint64_t tokens = 0;
  for (std::size_t index = 0; index < layer.sequences.size(); ++index) {
    const auto& item = layer.sequences[index];
    const auto item_tokens = item.input.input_quant.token_count;
    if (item.input_coordinator == nullptr ||
        item.output_coordinator == nullptr || item.transaction == nullptr ||
        item.sparse_query_bf16 == 0 || item.sparse_kv_bf16 == 0 ||
        item.sparse_output_bf16 == 0 || item_tokens == 0) {
      return Status::InvalidArgument(
          "DeepSeek dense attention packed work is incomplete");
    }
    tokens += item_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument(
          "DeepSeek dense attention token total overflows");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (item.input_coordinator == layer.sequences[prior].input_coordinator ||
          item.output_coordinator == layer.sequences[prior].output_coordinator ||
          item.transaction == layer.sequences[prior].transaction) {
        return Status::InvalidArgument(
            "DeepSeek dense attention packed state is aliased");
      }
    }
  }
  return tokens == descriptor.token_count
             ? Status::Ok()
             : Status::InvalidArgument(
                   "DeepSeek dense attention token total differs from plan");
}

}  // namespace

Result<DeepSeekBoundDenseAttentionWorkProvider>
DeepSeekBoundDenseAttentionWorkProvider::Create(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_sequences) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_sequences == 0) {
    return Status::InvalidArgument(
        "DeepSeek bound dense attention capacity is invalid");
  }
  DeepSeekBoundDenseAttentionWorkProvider provider;
  provider.owned_layers_ = owned_layers;
  provider.maximum_sequences_ = maximum_sequences;
  for (auto& bank : provider.banks_) {
    for (std::uint32_t layer = owned_layers.first_layer;
         layer <= owned_layers.last_layer; ++layer) {
      bank[layer].reserve(maximum_sequences);
    }
  }
  return provider;
}

Status DeepSeekBoundDenseAttentionWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekBoundDenseAttentionLayerWork> layers) {
  const auto expected = owned_layers_.last_layer - owned_layers_.first_layer + 1;
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 ||
      descriptor.sequence_count > maximum_sequences_ ||
      descriptor.phase == DeepSeekPlanPhase::kDrain ||
      layers.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek bound dense attention plan is invalid");
  }
  std::array<bool, 43> seen{};
  const auto pending = active_bank_ ^ 1U;
  for (const auto& layer : layers) {
    if (layer.layer < owned_layers_.first_layer ||
        layer.layer > owned_layers_.last_layer || seen[layer.layer]) {
      return Status::InvalidArgument(
          "DeepSeek bound dense attention layer is foreign or duplicated");
    }
    const auto status = validate_layer(layer, descriptor);
    if (!status.ok()) return status;
    banks_[pending][layer.layer].assign(layer.sequences.begin(),
                                        layer.sequences.end());
    seen[layer.layer] = true;
  }
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<std::span<const DeepSeekDenseAttentionStageSequenceWork>>
DeepSeekBoundDenseAttentionWorkProvider::resolve(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      command.kind != DeepSeekStageOperatorKind::kAttention ||
      command.layer < owned_layers_.first_layer ||
      command.layer > owned_layers_.last_layer) {
    return Status::FailedPrecondition(
        "DeepSeek dense attention work is not bound to this command and plan");
  }
  return std::span<const DeepSeekDenseAttentionStageSequenceWork>(
      banks_[active_bank_][command.layer]);
}

}  // namespace pih
