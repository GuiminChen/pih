#include "pih/model/deepseek_fixed_state_layout.h"

#include <algorithm>
#include <limits>

namespace pih {
namespace {

constexpr std::uint64_t kRecentBytes = kDeepSeekRecentStateBytes;
constexpr std::uint64_t kRatio4MainStateBytes = 8ULL * 1024ULL * 4ULL;
constexpr std::uint64_t kRatio4IndexStateBytes = 8ULL * 256ULL * 4ULL;
constexpr std::uint64_t kRatio128StateBytes = 128ULL * 512ULL * 4ULL;

DeepSeekRelativeStateSpan append(std::uint64_t& cursor, std::uint64_t bytes) {
  const DeepSeekRelativeStateSpan span{cursor, bytes};
  cursor += bytes;
  return span;
}

DeepSeekExpertArenaSpan resolve_span(std::uintptr_t base,
                                     DeepSeekRelativeStateSpan span) {
  return span.bytes == 0
             ? DeepSeekExpertArenaSpan{}
             : DeepSeekExpertArenaSpan{base + span.offset, span.bytes};
}

}  // namespace

Result<DeepSeekFixedStateLayout> DeepSeekFixedStateLayout::Build(
    std::span<const std::uint32_t> owned_main_layers, bool include_dspark) {
#if defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (include_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark fixed state is not present in this model bundle");
  }
#endif
  if (owned_main_layers.empty() && !include_dspark) {
    return Status::InvalidArgument("DeepSeek fixed state layout is empty");
  }
  for (std::size_t index = 0; index < owned_main_layers.size(); ++index) {
    if (owned_main_layers[index] > 42 ||
        (index != 0 && owned_main_layers[index - 1] >= owned_main_layers[index])) {
      return Status::InvalidArgument(
          "DeepSeek fixed state main layers are not canonical");
    }
  }
  DeepSeekFixedStateLayout layout;
  layout.descriptors_.reserve(owned_main_layers.size());
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  layout.dspark_descriptors_.reserve(include_dspark
                                         ? kDeepSeekDsparkStageCount
                                         : 0U);
#endif
  std::uint64_t cursor = 0;
  for (const auto layer : owned_main_layers) {
    DeepSeekFixedLayerStateDescriptor descriptor;
    descriptor.layer_id = layer;
    descriptor.kind = layer < 2 ? DeepSeekFixedLayerKind::kRecentOnly
                      : layer % 2 == 0 ? DeepSeekFixedLayerKind::kRatio4
                                       : DeepSeekFixedLayerKind::kRatio128;
    descriptor.recent_bf16 = append(cursor, kRecentBytes);
    if (descriptor.kind == DeepSeekFixedLayerKind::kRatio4) {
      descriptor.main_kv_state_f32 = append(cursor, kRatio4MainStateBytes);
      descriptor.main_score_state_f32 = append(cursor, kRatio4MainStateBytes);
      descriptor.index_kv_state_f32 = append(cursor, kRatio4IndexStateBytes);
      descriptor.index_score_state_f32 = append(cursor, kRatio4IndexStateBytes);
    } else if (descriptor.kind == DeepSeekFixedLayerKind::kRatio128) {
      descriptor.main_kv_state_f32 = append(cursor, kRatio128StateBytes);
      descriptor.main_score_state_f32 = append(cursor, kRatio128StateBytes);
    }
    layout.descriptors_.push_back(descriptor);
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (include_dspark) {
    for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
      auto stage = deepseek_dspark_stage_id(index);
      if (!stage.ok()) return stage.status();
      DeepSeekDsparkFixedStageStateDescriptor descriptor;
      descriptor.stage = *stage;
      descriptor.recent_bf16 = append(cursor, kRecentBytes);
      layout.dspark_descriptors_.push_back(descriptor);
    }
  }
#endif
  layout.total_bytes_ = cursor;
  return layout;
}

Result<DeepSeekFixedLayerDeviceView> DeepSeekFixedStateLayout::Resolve(
    std::uint32_t layer_id, DeepSeekExpertArenaSpan bank) const {
  if (bank.address == 0 || bank.bytes < total_bytes_ ||
      bank.address > std::numeric_limits<std::uintptr_t>::max() - total_bytes_) {
    return Status::InvalidArgument("DeepSeek fixed state bank is invalid");
  }
  const auto found = std::find_if(
      descriptors_.begin(), descriptors_.end(),
      [layer_id](const auto& descriptor) {
        return descriptor.layer_id == layer_id;
      });
  if (found == descriptors_.end()) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state layer is not stage-owned");
  }
  return DeepSeekFixedLayerDeviceView{
      found->layer_id,
      found->kind,
      resolve_span(bank.address, found->recent_bf16),
      resolve_span(bank.address, found->main_kv_state_f32),
      resolve_span(bank.address, found->main_score_state_f32),
      resolve_span(bank.address, found->index_kv_state_f32),
      resolve_span(bank.address, found->index_score_state_f32)};
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekDsparkFixedStageDeviceView>
DeepSeekFixedStateLayout::ResolveDspark(
    DeepSeekDsparkStageId stage, DeepSeekExpertArenaSpan bank) const {
  if (!is_valid_deepseek_dspark_stage(stage) || bank.address == 0 ||
      bank.bytes < total_bytes_ ||
      bank.address > std::numeric_limits<std::uintptr_t>::max() -
                         total_bytes_) {
    return Status::InvalidArgument(
        "DeepSeek DSpark fixed state bank is invalid");
  }
  const auto found = std::find_if(
      dspark_descriptors_.begin(), dspark_descriptors_.end(),
      [stage](const auto& descriptor) {
        return descriptor.stage == stage;
      });
  if (found == dspark_descriptors_.end()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark fixed state is not stage-owned");
  }
  return DeepSeekDsparkFixedStageDeviceView{
      found->stage, resolve_span(bank.address, found->recent_bf16)};
}
#endif

}  // namespace pih
