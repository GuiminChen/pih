#include "pih/model/deepseek_bound_attention_work_provider.h"

namespace pih {
namespace {

bool same_descriptor(const DeepSeekPipelinePlanDescriptor& a,
                     const DeepSeekPipelinePlanDescriptor& b) {
  return a.engine_epoch == b.engine_epoch && a.plan_sequence == b.plan_sequence &&
         a.phase == b.phase && a.token_count == b.token_count &&
         a.sequence_count == b.sequence_count;
}

bool valid_range(DeepSeekStageRange range) {
  return range.first_layer <= range.last_layer && range.last_layer <= 42;
}

bool valid_decode(const DeepSeekDecodeAttentionWork& work,
                  std::uint32_t token_count) {
  return work.recent_writer != nullptr && work.update_coordinator != nullptr &&
         work.coordinator != nullptr && work.transaction != nullptr &&
         work.attention.query_positions.size() == token_count;
}

bool valid_chunk(const DeepSeekChunkAttentionWork& work,
                 std::uint32_t token_count) {
  return work.chunk_coordinator != nullptr &&
         work.attention_coordinator != nullptr && work.transaction != nullptr &&
         work.submission.attention.query_positions.size() == token_count &&
         !work.submission.recent.empty() &&
         (work.submission.attention.kind == DeepSeekCompressedAttentionKind::kRecentOnly
              ? work.submission.updates.empty() : !work.submission.updates.empty());
}

}  // namespace

Result<DeepSeekBoundDecodeAttentionWorkProvider>
DeepSeekBoundDecodeAttentionWorkProvider::Create(
    DeepSeekStageRange owned_layers) {
  if (!valid_range(owned_layers)) {
    return Status::InvalidArgument(
        "DeepSeek bound decode attention layer range is invalid");
  }
  DeepSeekBoundDecodeAttentionWorkProvider provider;
  provider.owned_layers_ = owned_layers;
  return provider;
}

Status DeepSeekBoundDecodeAttentionWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekBoundDecodeAttentionLayerWork> layers) {
  const auto expected = owned_layers_.last_layer - owned_layers_.first_layer + 1;
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.phase != DeepSeekPlanPhase::kDecode ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 || descriptor.sequence_count > 4096 ||
      (layers.size() != expected && layers.size() != expected * descriptor.sequence_count)) {
    return Status::InvalidArgument(
        "DeepSeek bound decode attention plan is invalid");
  }
  std::array<std::uint32_t, 43> seen{};
  const auto pending = active_bank_ ^ 1U;
  for (auto& batch : banks_[pending]) batch.clear();
  for (const auto& layer : layers) {
    if (layer.layer < owned_layers_.first_layer ||
        layer.layer > owned_layers_.last_layer || seen[layer.layer] >= (layers.size() == expected ? 1U : descriptor.sequence_count) ||
        !valid_decode(layer.work, layers.size() == expected ? descriptor.token_count : descriptor.token_count / descriptor.sequence_count)) {
      return Status::InvalidArgument(
          "DeepSeek bound decode attention layer work is invalid");
    }
    for (const auto& previous : banks_[pending][layer.layer]) {
      if (previous.transaction == layer.work.transaction || previous.coordinator == layer.work.coordinator)
        return Status::InvalidArgument("DeepSeek packed attention lane is duplicated");
    }
    banks_[pending][layer.layer].push_back(layer.work);
    ++seen[layer.layer];
  }
  for (auto layer = owned_layers_.first_layer; layer <= owned_layers_.last_layer; ++layer) {
    if (seen[layer] != (layers.size() == expected ? 1U : descriptor.sequence_count)) return Status::InvalidArgument("DeepSeek packed attention coverage is incomplete");
  }
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<const DeepSeekDecodeAttentionWork*>
DeepSeekBoundDecodeAttentionWorkProvider::resolve(
    std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      layer < owned_layers_.first_layer || layer > owned_layers_.last_layer) {
    return Status::FailedPrecondition(
        "DeepSeek decode attention work is not bound to this layer and plan");
  }
  return &banks_[active_bank_][layer].front();
}

Status DeepSeekBoundDecodeAttentionWorkProvider::reset_completed() {
  if (!descriptor_.has_value()) return Status::Ok();
  for (auto layer = owned_layers_.first_layer;
       layer <= owned_layers_.last_layer; ++layer) {
    for (const auto& work : banks_[active_bank_][layer]) {
    auto* coordinator = work.coordinator;
    if (coordinator == nullptr) return Status::Internal("DeepSeek completed decode coordinator is missing");
    auto status = coordinator->reset();
    if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

Result<DeepSeekBoundChunkAttentionWorkProvider>
DeepSeekBoundChunkAttentionWorkProvider::Create(
    DeepSeekStageRange owned_layers) {
  if (!valid_range(owned_layers)) {
    return Status::InvalidArgument(
        "DeepSeek bound chunk attention layer range is invalid");
  }
  DeepSeekBoundChunkAttentionWorkProvider provider;
  provider.owned_layers_ = owned_layers;
  return provider;
}

Status DeepSeekBoundChunkAttentionWorkProvider::bind(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::span<const DeepSeekBoundChunkAttentionLayerWork> layers) {
  const bool phase = descriptor.phase == DeepSeekPlanPhase::kPrefill ||
                     descriptor.phase == DeepSeekPlanPhase::kVerify;
  const auto expected = owned_layers_.last_layer - owned_layers_.first_layer + 1;
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 || !phase ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 || descriptor.sequence_count > 4096 ||
      (layers.size() != expected && layers.size() != expected * descriptor.sequence_count)) {
    return Status::InvalidArgument(
        "DeepSeek bound chunk attention plan is invalid");
  }
  std::array<std::uint32_t, 43> seen{};
  const auto pending = active_bank_ ^ 1U;
  for (auto& batch : banks_[pending]) batch.clear();
  for (const auto& layer : layers) {
    if (layer.layer < owned_layers_.first_layer ||
        layer.layer > owned_layers_.last_layer || seen[layer.layer] >= (layers.size() == expected ? 1U : descriptor.sequence_count) ||
        !valid_chunk(layer.work, layers.size() == expected ? descriptor.token_count : descriptor.token_count / descriptor.sequence_count)) {
      return Status::InvalidArgument(
          "DeepSeek bound chunk attention layer work is invalid");
    }
    for (const auto& previous : banks_[pending][layer.layer]) {
      if (previous.transaction == layer.work.transaction || previous.attention_coordinator == layer.work.attention_coordinator)
        return Status::InvalidArgument("DeepSeek packed attention lane is duplicated");
    }
    banks_[pending][layer.layer].push_back(layer.work);
    ++seen[layer.layer];
  }
  for (auto layer = owned_layers_.first_layer; layer <= owned_layers_.last_layer; ++layer) {
    if (seen[layer] != (layers.size() == expected ? 1U : descriptor.sequence_count)) return Status::InvalidArgument("DeepSeek packed attention coverage is incomplete");
  }
  active_bank_ = pending;
  descriptor_ = descriptor;
  return Status::Ok();
}

Result<const DeepSeekChunkAttentionWork*>
DeepSeekBoundChunkAttentionWorkProvider::resolve(
    std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) {
  if (!descriptor_.has_value() || !same_descriptor(*descriptor_, plan) ||
      layer < owned_layers_.first_layer || layer > owned_layers_.last_layer) {
    return Status::FailedPrecondition(
        "DeepSeek chunk attention work is not bound to this layer and plan");
  }
  return &banks_[active_bank_][layer].front();
}

Status DeepSeekBoundChunkAttentionWorkProvider::reset_completed() {
  if (!descriptor_.has_value()) return Status::Ok();
  for (auto layer = owned_layers_.first_layer;
       layer <= owned_layers_.last_layer; ++layer) {
    for (const auto& work : banks_[active_bank_][layer]) {
    auto* coordinator = work.attention_coordinator;
    if (coordinator == nullptr) return Status::Internal("DeepSeek completed chunk coordinator is missing");
    auto status = coordinator->reset();
    if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

Result<std::span<const DeepSeekDecodeAttentionWork>>
DeepSeekBoundDecodeAttentionWorkProvider::resolve_batch(
    std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) {
  auto first = resolve(layer, plan);
  if (!first.ok()) return first.status();
  return std::span<const DeepSeekDecodeAttentionWork>(banks_[active_bank_][layer]);
}
Result<std::span<const DeepSeekChunkAttentionWork>>
DeepSeekBoundChunkAttentionWorkProvider::resolve_batch(
    std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) {
  auto first = resolve(layer, plan);
  if (!first.ok()) return first.status();
  return std::span<const DeepSeekChunkAttentionWork>(banks_[active_bank_][layer]);
}
}  // namespace pih
