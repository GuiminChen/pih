#pragma once

#include <functional>
#include <string_view>

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_stage_identity.h"
#endif
#include "pih/model/deepseek_resident_weight_arena.h"

namespace pih {

struct DeepSeekAttentionWeightBindings final {
  std::uintptr_t wq_a_fp8 = 0, wq_a_scale_ue8m0 = 0;
  std::uintptr_t q_norm_bf16 = 0;
  std::uintptr_t wq_b_fp8 = 0, wq_b_scale_ue8m0 = 0;
  std::uintptr_t wkv_fp8 = 0, wkv_scale_ue8m0 = 0;
  std::uintptr_t kv_norm_bf16 = 0;
  std::uintptr_t attention_sink_f32 = 0;
  std::uintptr_t wo_a_fp8 = 0, wo_a_scale_ue8m0 = 0;
  std::uintptr_t wo_b_fp8 = 0, wo_b_scale_ue8m0 = 0;
  std::uint64_t generation = 0;

  using Resolver = std::function<Result<TensorView>(std::string_view)>;
  static Result<DeepSeekAttentionWeightBindings> Resolve(
      std::uint32_t layer, const Resolver& resolver);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  static Result<DeepSeekAttentionWeightBindings> ResolveDspark(
      DeepSeekDsparkStageId stage, const Resolver& resolver);
#endif
  static Result<DeepSeekAttentionWeightBindings> Resolve(
      std::uint32_t layer, const DeepSeekResidentWeightArena& arena);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  static Result<DeepSeekAttentionWeightBindings> ResolveDspark(
      DeepSeekDsparkStageId stage,
      const DeepSeekResidentWeightArena& arena);
#endif
};

}  // namespace pih
