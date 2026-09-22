#pragma once

#include <array>
#include <optional>
#include <span>

#include "pih/model/deepseek_router_stage_backend.h"

namespace pih {

struct DeepSeekBoundHashRouterWork final {
  std::uint32_t layer = 0;
  std::span<const std::uint32_t> token_ids;
  std::span<const float> raw_scores;
  std::uint32_t vocabulary_size = 0;
  std::span<const std::uint16_t> token_to_experts;
  DeepSeekRouteScratchArena* scratch = nullptr;
  DeepSeekHashRouterCoordinator* coordinator = nullptr;
  DeepSeekHashRouterSubmission submission;
};

struct DeepSeekBoundLearnedRouterWork final {
  std::uint32_t layer = 0;
  DeepSeekLearnedRouterCoordinator* coordinator = nullptr;
  DeepSeekLearnedRouterSubmission submission;
};

class DeepSeekBoundRouterStageWorkProvider final
    : public DeepSeekRouterStageWorkProvider {
 public:
  static Result<DeepSeekBoundRouterStageWorkProvider> Create(
      DeepSeekStageRange owned_layers, DeepSeekExpertPlanStore& store);

  Status bind(const DeepSeekPipelinePlanDescriptor& descriptor,
              std::span<const DeepSeekBoundHashRouterWork> hash_work,
              std::span<const DeepSeekBoundLearnedRouterWork> learned_work);
  void clear() noexcept;

  DeepSeekExpertPlanProvider* plan_provider() noexcept override {
    return store_;
  }
  Result<DeepSeekHashRouterStageWork> resolve_hash(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekLearnedRouterStageWork> resolve_learned(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  DeepSeekStageRange owned_layers_;
  DeepSeekExpertPlanStore* store_ = nullptr;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
  std::array<std::optional<DeepSeekHashRouterStageWork>, 3> hash_;
  std::array<std::optional<DeepSeekLearnedRouterStageWork>, 43> learned_;
};

}  // namespace pih
