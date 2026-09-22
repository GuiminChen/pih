#include "pih/model/deepseek_rank_router_factory_set.h"

namespace pih {

Result<DeepSeekRankRouterFactorySet>
DeepSeekRankRouterFactorySet::Create(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size, const DeepSeekWeightByteSource& source,
    DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations) {
  auto hash_tables = DeepSeekHashRouterTableLoader::Load(
      owned_layers, vocabulary_size, source);
  if (!hash_tables.ok()) return hash_tables.status();
  auto biases = DeepSeekLearnedRouterBiasLoader::Load(owned_layers, source);
  if (!biases.ok()) return biases.status();
  auto hash_router = DeepSeekHashRouterWorkFactory::CreateProjected(
      owned_layers, maximum_tokens, vocabulary_size,
      std::move(*hash_tables), store, operations);
  if (!hash_router.ok()) return hash_router.status();
  auto learned_router = DeepSeekLearnedRouterWorkFactory::Create(
      owned_layers, maximum_tokens, std::move(*biases), store, operations);
  if (!learned_router.ok()) return learned_router.status();
  return DeepSeekRankRouterFactorySet(
      std::move(*hash_router), std::move(*learned_router));
}

}  // namespace pih
