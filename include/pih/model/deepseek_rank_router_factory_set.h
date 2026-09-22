#pragma once

#include "pih/model/deepseek_hash_router_table_loader.h"
#include "pih/model/deepseek_learned_router_bias_loader.h"

namespace pih {

class DeepSeekRankRouterFactorySet final {
 public:
  static Result<DeepSeekRankRouterFactorySet> Create(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size, const DeepSeekWeightByteSource& source,
      DeepSeekExpertPlanStore& store,
      DeepSeekLearnedRouterOperations& operations);

  DeepSeekRankRouterFactorySet(const DeepSeekRankRouterFactorySet&) = delete;
  DeepSeekRankRouterFactorySet& operator=(
      const DeepSeekRankRouterFactorySet&) = delete;
  DeepSeekRankRouterFactorySet(DeepSeekRankRouterFactorySet&&) noexcept =
      default;
  DeepSeekRankRouterFactorySet& operator=(
      DeepSeekRankRouterFactorySet&&) noexcept = default;

  [[nodiscard]] DeepSeekHashRouterWorkFactory& hash_router() noexcept {
    return hash_router_;
  }
  [[nodiscard]] DeepSeekLearnedRouterWorkFactory& learned_router() noexcept {
    return learned_router_;
  }

 private:
  DeepSeekRankRouterFactorySet(
      DeepSeekHashRouterWorkFactory hash_router,
      DeepSeekLearnedRouterWorkFactory learned_router) noexcept
      : hash_router_(std::move(hash_router)),
        learned_router_(std::move(learned_router)) {}

  DeepSeekHashRouterWorkFactory hash_router_;
  DeepSeekLearnedRouterWorkFactory learned_router_;
};

}  // namespace pih
