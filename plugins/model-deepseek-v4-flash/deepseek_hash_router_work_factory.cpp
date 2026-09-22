#include "pih/model/deepseek_hash_router_work_factory.h"

#include <algorithm>
#include <limits>

namespace pih {

Result<DeepSeekHashRouterWorkFactory>
DeepSeekHashRouterWorkFactory::Create(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size,
    std::vector<DeepSeekHashRouterLayerTable> tables) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43 || maximum_tokens == 0 ||
      vocabulary_size == 0 ||
      vocabulary_size > std::numeric_limits<std::size_t>::max() /
                            DeepSeekExpertSubwavePlan::kRoutesPerToken) {
    return Status::InvalidArgument(
        "DeepSeek hash router work factory identity is invalid");
  }
  const auto last_hash_layer = std::min(owned_layers.last_layer, 2U);
  const auto expected = owned_layers.first_layer <= last_hash_layer
                            ? last_hash_layer - owned_layers.first_layer + 1
                            : 0U;
  if (tables.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek hash router tables do not cover owned hash layers");
  }
  DeepSeekHashRouterWorkFactory result;
  result.owned_layers_ = owned_layers;
  result.maximum_tokens_ = maximum_tokens;
  result.vocabulary_size_ = vocabulary_size;
  result.owned_hash_layer_count_ = expected;
  std::array<bool, 3> seen{};
  const auto table_size =
      static_cast<std::size_t>(vocabulary_size) *
      DeepSeekExpertSubwavePlan::kRoutesPerToken;
  for (auto& table : tables) {
    if (table.layer < owned_layers.first_layer ||
        table.layer > last_hash_layer || seen[table.layer] ||
        table.token_to_experts.size() != table_size) {
      return Status::InvalidArgument(
          "DeepSeek hash router table is invalid or duplicated");
    }
    seen[table.layer] = true;
    result.tables_[table.layer] =
        std::make_shared<const std::vector<std::uint16_t>>(
            std::move(table.token_to_experts));
    auto scratch = DeepSeekRouteScratchArena::Create(maximum_tokens);
    if (!scratch.ok()) return scratch.status();
    result.scratch_[table.layer] = std::move(*scratch);
  }
  return result;
}

Result<DeepSeekHashRouterWorkFactory>
DeepSeekHashRouterWorkFactory::CreateProjected(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size,
    std::vector<DeepSeekHashRouterLayerTable> tables,
    DeepSeekExpertPlanStore& store,
    DeepSeekLearnedRouterOperations& operations) {
  auto result = Create(owned_layers, maximum_tokens, vocabulary_size,
                       std::move(tables));
  if (!result.ok()) return result.status();
  for (std::uint32_t layer = 0; layer < result->scratch_.size(); ++layer) {
    if (!result->scratch_[layer].has_value()) continue;
    auto coordinator = DeepSeekHashRouterCoordinator::Create(
        maximum_tokens, *result->scratch_[layer], store, operations);
    if (!coordinator.ok()) return coordinator.status();
    result->coordinators_[layer] = std::move(*coordinator);
  }
  return result;
}

Status DeepSeekHashRouterWorkFactory::append_plan_work(
    std::span<const std::uint32_t> token_ids,
    std::vector<DeepSeekHashRouterLayerScores> layer_scores,
    DeepSeekRankComputeWorkBuilder& builder) {
  if (token_ids.empty() || token_ids.size() > maximum_tokens_ ||
      layer_scores.size() != owned_hash_layer_count_ ||
      std::ranges::any_of(token_ids, [this](std::uint32_t token) {
        return token >= vocabulary_size_;
      })) {
    return Status::InvalidArgument(
        "DeepSeek hash router plan tokens or layer count is invalid");
  }
  std::array<bool, 3> seen{};
  for (auto& scores : layer_scores) {
    if (scores.layer >= 3 || tables_[scores.layer] == nullptr ||
        seen[scores.layer] ||
        scores.raw_scores.size() != token_ids.size() *
            DeepSeekExpertSubwavePlan::kExpertCount) {
      return Status::InvalidArgument(
          "DeepSeek hash router plan scores are invalid or duplicated");
    }
    seen[scores.layer] = true;
    const auto status = builder.add_hash_router_with_shared_table(
        scores.layer,
        std::vector<std::uint32_t>(token_ids.begin(), token_ids.end()),
        std::move(scores.raw_scores), vocabulary_size_,
        tables_[scores.layer], &*scratch_[scores.layer]);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status DeepSeekHashRouterWorkFactory::append_projected_plan_work(
    std::span<const std::uint32_t> token_ids,
    std::vector<DeepSeekHashRouterLayerPlanWork> layers,
    DeepSeekRankComputeWorkBuilder& builder) {
  if (token_ids.empty() || token_ids.size() > maximum_tokens_ ||
      layers.size() != owned_hash_layer_count_ ||
      std::ranges::any_of(token_ids, [this](std::uint32_t token) {
        return token >= vocabulary_size_;
      })) {
    return Status::InvalidArgument(
        "DeepSeek projected hash router plan capacity is invalid");
  }
  std::array<bool, 3> seen{};
  for (auto& work : layers) {
    const auto layer = work.submission.layer;
    const auto launch_status = validate_deepseek_router_bf16_gemm_launch(
        {work.submission.input_bf16, work.submission.weight_bf16,
         work.submission.scores_f32, work.submission.error_flag_u32,
         work.submission.stream, work.submission.token_count,
         DeepSeekLearnedRouterCoordinator::kExpertCount,
         DeepSeekLearnedRouterCoordinator::kHiddenSize});
    if (layer >= coordinators_.size() ||
        !coordinators_[layer].has_value() || tables_[layer] == nullptr ||
        seen[layer] || work.submission.token_count != token_ids.size() ||
        work.submission.completion_event == 0 ||
        !launch_status.ok() || work.host_staging == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek projected hash router layer is invalid or duplicated");
    }
    seen[layer] = true;
    auto status = builder.add_projected_hash_router_with_shared_table(
        layer, std::vector<std::uint32_t>(token_ids.begin(), token_ids.end()),
        &*coordinators_[layer], work.submission, vocabulary_size_,
        tables_[layer], std::move(work.host_staging));
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

}  // namespace pih
