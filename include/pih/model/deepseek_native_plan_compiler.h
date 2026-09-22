#pragma once

#include <vector>

#include "pih/model/deepseek_native_rank_compute_plan.h"
#include "pih/model/deepseek_rank_plan_compiler.h"

namespace pih {

class DeepSeekNativePlanCompiler final {
 public:
  static Result<DeepSeekNativePlanCompiler> Create(
      DeepSeekPipelinePlan topology,
      std::vector<DeepSeekRankPlanCompiler> rank_compilers);

  Result<DeepSeekNativeRankComputePlan> compile(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRankPlanCompilerInput> rank_inputs);

  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return topology_.world_size();
  }
  [[nodiscard]] const DeepSeekPipelinePlan& topology() const noexcept {
    return topology_;
  }
  std::vector<DeepSeekRankPlanCompiler> release_rank_compilers() && {
    return std::move(rank_compilers_);
  }

 private:
  DeepSeekPipelinePlan topology_;
  std::vector<DeepSeekRankPlanCompiler> rank_compilers_;
};

}  // namespace pih
