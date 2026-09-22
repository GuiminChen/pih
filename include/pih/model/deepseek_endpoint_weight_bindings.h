#pragma once

#include <functional>
#include <string_view>

#include "pih/model/deepseek_resident_weight_arena.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

struct DeepSeekEndpointWeightBindings final {
  std::uintptr_t embedding_weight_bf16 = 0;
  std::uintptr_t hc_head_fn_f32 = 0;
  std::uintptr_t hc_head_scale_f32 = 0;
  std::uintptr_t hc_head_base_f32 = 0;
  std::uintptr_t norm_weight_bf16 = 0;
  std::uintptr_t head_weight_bf16 = 0;
  std::uint64_t generation = 0;

  using Resolver = std::function<Result<TensorView>(std::string_view)>;
  static Result<DeepSeekEndpointWeightBindings> Resolve(
      DeepSeekStagePlan stage, const Resolver& resolver);
  static Result<DeepSeekEndpointWeightBindings> Resolve(
      DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena);
};

}  // namespace pih
