#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "pih/model/deepseek_deferred_rank_compute_plan_compiler.h"
#include "pih/model/deepseek_rank_plan_compiler.h"

namespace pih {

class DeepSeekDeferredRankPlanInputAssembler {
 public:
  virtual ~DeepSeekDeferredRankPlanInputAssembler() = default;
  virtual Result<DeepSeekRankPlanCompilerInput> assemble(
      std::uintptr_t incoming_activation_bf16) = 0;
};

class DeepSeekDeferredNativePlanCompiler final
    : public DeepSeekDeferredRankComputePlanCompiler {
 public:
  static Result<DeepSeekDeferredNativePlanCompiler> Create(
      DeepSeekPipelinePlan topology,
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRankPlanCompiler> rank_compilers,
      std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
          input_assemblers);

  [[nodiscard]] std::uint32_t world_size() const noexcept override {
    return topology_.world_size();
  }
  Result<DeepSeekRankComputePlanWork> compile(
      std::uint32_t rank,
      std::uintptr_t incoming_activation_bf16) override;

 private:
  DeepSeekPipelinePlan topology_;
  DeepSeekPipelinePlanDescriptor descriptor_;
  std::vector<DeepSeekRankPlanCompiler> rank_compilers_;
  std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
      input_assemblers_;
  std::vector<bool> compiled_;
  std::optional<Status> failure_;
};

}  // namespace pih
