#include "pih/model/deepseek_bound_mhc_stage_work_provider.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& a,
                     const DeepSeekPipelinePlanDescriptor& b) {
  return a.engine_epoch == b.engine_epoch && a.plan_sequence == b.plan_sequence &&
         a.phase == b.phase && a.token_count == b.token_count &&
         a.sequence_count == b.sequence_count;
}

Status copy_branch(
    std::span<const DeepSeekBoundMhcLayerWork> source,
    DeepSeekMhcBranchKind kind, DeepSeekStageRange owned,
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::array<std::vector<DeepSeekMhcStageSequenceWork>, 43>& target) {
  const auto expected = owned.last_layer - owned.first_layer + 1;
  if (source.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek bound mHC branch does not cover every owned layer");
  }
  std::array<bool, 43> seen{};
  for (const auto& layer : source) {
    if (layer.layer < owned.first_layer || layer.layer > owned.last_layer ||
        seen[layer.layer] || layer.sequences.size() != descriptor.sequence_count) {
      return Status::InvalidArgument(
          "DeepSeek bound mHC layer work is invalid or duplicated");
    }
    for (std::size_t index = 0; index < layer.sequences.size(); ++index) {
      const auto& item = layer.sequences[index];
      if (item.executor == nullptr || item.transaction == nullptr ||
          item.submission.kind != kind ||
          item.submission.layer_id != layer.layer ||
          item.submission.token_count != descriptor.token_count) {
        return Status::InvalidArgument(
            "DeepSeek bound mHC packed work is inconsistent");
      }
      for (std::size_t prior = 0; prior < index; ++prior) {
        if (item.executor == layer.sequences[prior].executor ||
            item.transaction == layer.sequences[prior].transaction) {
          return Status::InvalidArgument(
              "DeepSeek bound mHC packed state is aliased");
        }
      }
    }
    target[layer.layer].assign(layer.sequences.begin(), layer.sequences.end());
    seen[layer.layer] = true;
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekBoundMhcStageWorkProvider>
DeepSeekBoundMhcStageWorkProvider::Create(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_sequences) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_sequences == 0) {
    return Status::InvalidArgument(
        "DeepSeek bound mHC provider capacity is invalid");
  }
  DeepSeekBoundMhcStageWorkProvider provider;
  provider.owned_layers_ = owned_layers;
  provider.maximum_sequences_ = maximum_sequences;
  for (auto& bank : provider.banks_) {
    for (auto& branch : bank) {
      for (std::uint32_t layer = owned_layers.first_layer;
           layer <= owned_layers.last_layer; ++layer) {
        branch[layer].reserve(maximum_sequences);
      }
    }
  }
  return provider;
}

Status DeepSeekBoundMhcStageWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekBoundMhcLayerWork> attention,
    std::span<const DeepSeekBoundMhcLayerWork> feed_forward) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 ||
      descriptor.sequence_count > maximum_sequences_ ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument("DeepSeek bound mHC plan is invalid");
  }
  const auto pending = active_bank_ ^ 1U;
  auto status = copy_branch(attention, DeepSeekMhcBranchKind::kAttention,
                            owned_layers_, descriptor, banks_[pending][0]);
  if (!status.ok()) return status;
  status = copy_branch(feed_forward, DeepSeekMhcBranchKind::kFeedForward,
                       owned_layers_, descriptor, banks_[pending][1]);
  if (!status.ok()) return status;
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<std::span<const DeepSeekMhcStageSequenceWork>>
DeepSeekBoundMhcStageWorkProvider::resolve(
    const DeepSeekStageOperatorCommand& command,
    const DeepSeekPipelinePlanDescriptor& plan) {
  const bool attention = command.kind == DeepSeekStageOperatorKind::kAttention;
  const bool feed_forward = command.kind == DeepSeekStageOperatorKind::kMoe;
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      (!attention && !feed_forward) ||
      command.layer < owned_layers_.first_layer ||
      command.layer > owned_layers_.last_layer) {
    return Status::FailedPrecondition(
        "DeepSeek mHC work is not bound to this command and plan");
  }
  return std::span<const DeepSeekMhcStageSequenceWork>(
      banks_[active_bank_][feed_forward ? 1 : 0][command.layer]);
}

}  // namespace pih
