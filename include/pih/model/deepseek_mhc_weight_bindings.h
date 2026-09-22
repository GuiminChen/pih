#pragma once

#include <functional>
#include <string_view>

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_stage_identity.h"
#endif
#include "pih/model/deepseek_resident_weight_arena.h"

namespace pih {

struct DeepSeekMhcWeightBindings final {
  std::uintptr_t attention_norm_bf16 = 0;
  std::uintptr_t attention_fn_f32 = 0;
  std::uintptr_t attention_scale_f32 = 0;
  std::uintptr_t attention_base_f32 = 0;
  std::uintptr_t feed_forward_norm_bf16 = 0;
  std::uintptr_t feed_forward_fn_f32 = 0;
  std::uintptr_t feed_forward_scale_f32 = 0;
  std::uintptr_t feed_forward_base_f32 = 0;
  std::uint64_t generation = 0;

  using Resolver = std::function<Result<TensorView>(std::string_view)>;
  static Result<DeepSeekMhcWeightBindings> Resolve(
      std::uint32_t layer, const Resolver& resolver);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  static Result<DeepSeekMhcWeightBindings> ResolveDspark(
      DeepSeekDsparkStageId stage, const Resolver& resolver);
#endif
  static Result<DeepSeekMhcWeightBindings> Resolve(
      std::uint32_t layer, const DeepSeekResidentWeightArena& arena);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  static Result<DeepSeekMhcWeightBindings> ResolveDspark(
      DeepSeekDsparkStageId stage,
      const DeepSeekResidentWeightArena& arena);
#endif
};

}  // namespace pih
