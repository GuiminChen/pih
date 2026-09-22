#include "pih/model/deepseek_expert_compute_arena.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekExpertComputeArenaLayout>
DeepSeekExpertComputeArenaLayout::Create(std::uint32_t maximum_tokens) {
  if (maximum_tokens == 0 || maximum_tokens > kMaximumTokens) {
    return Status::InvalidArgument(
        "DeepSeek expert compute token capacity is invalid");
  }
  DeepSeekExpertComputeArenaLayout layout;
  layout.maximum_tokens_ = maximum_tokens;
  const auto tokens = static_cast<std::uint64_t>(maximum_tokens);
  const auto append_region = [](std::uint64_t cursor,
                                std::uint64_t bytes) -> Result<Region> {
    auto offset = checked_align_up_u64(cursor, kAlignment);
    if (!offset.ok()) return offset.status();
    auto end = checked_add_u64(*offset, bytes);
    if (!end.ok()) return end.status();
    return Region{*offset, bytes};
  };
  const auto end_of = [](const Region& region) {
    return region.offset + region.bytes;
  };
  auto input = append_region(0, tokens * 4096U * 2U);
  if (!input.ok()) return input.status();
  layout.route_input_output_ = *input;
  auto activation = append_region(end_of(*input), tokens * 4096U);
  if (!activation.ok()) return activation.status();
  layout.activation_ = *activation;
  auto scales = append_region(end_of(*activation), tokens * (4096U / 128U));
  if (!scales.ok()) return scales.status();
  layout.activation_scales_ = *scales;
  auto gate = append_region(end_of(*scales), tokens * 2048U * 2U);
  if (!gate.ok()) return gate.status();
  layout.gate_middle_ = *gate;
  auto up = append_region(end_of(*gate), tokens * 2048U * 2U);
  if (!up.ok()) return up.status();
  layout.up_ = *up;
  auto weights = append_region(end_of(*up), tokens * 4U);
  if (!weights.ok()) return weights.status();
  layout.route_weights_ = *weights;
  auto indices = append_region(end_of(*weights), tokens * 4U);
  if (!indices.ok()) return indices.status();
  layout.token_indices_ = *indices;
  auto error = append_region(end_of(*indices), kAlignment);
  if (!error.ok()) return error.status();
  layout.error_flag_ = *error;
  layout.required_bytes_ = end_of(*error);
  return layout;
}

Result<DeepSeekExpertComputeArena> DeepSeekExpertComputeArenaLayout::bind(
    std::uintptr_t base, std::uint64_t backing_bytes) const {
  if (maximum_tokens_ == 0 || base == 0 || base % kAlignment != 0 ||
      backing_bytes < required_bytes_ ||
      required_bytes_ > std::numeric_limits<std::uintptr_t>::max() ||
      base > std::numeric_limits<std::uintptr_t>::max() - required_bytes_) {
    return Status::InvalidArgument(
        "DeepSeek expert compute arena backing is invalid");
  }
  const auto bind_region = [base](const Region& region) {
    return DeepSeekExpertArenaSpan{
        base + static_cast<std::uintptr_t>(region.offset), region.bytes};
  };
  const auto input_output = bind_region(route_input_output_);
  return DeepSeekExpertComputeArena{
      .route_input_bf16 = input_output,
      .expert_output_bf16 = input_output,
      .activation_e4m3 = bind_region(activation_),
      .activation_scale_bits = bind_region(activation_scales_),
      .gate_or_middle_bf16 = bind_region(gate_middle_),
      .up_bf16 = bind_region(up_),
      .route_weights_f32 = bind_region(route_weights_),
      .token_indices_u32 = bind_region(token_indices_),
      .error_flag_u32 = bind_region(error_flag_),
  };
}

}  // namespace pih
