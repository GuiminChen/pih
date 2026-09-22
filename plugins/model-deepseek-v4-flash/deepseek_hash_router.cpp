#include "pih/model/deepseek_hash_router.h"

#include <array>
#include <cmath>

namespace pih {
namespace {

float hash_router_base(float raw) {
  const auto softplus = raw > 0.0F
                            ? raw + std::log1p(std::exp(-raw))
                            : std::log1p(std::exp(raw));
  return std::sqrt(softplus);
}

Status route_into(
    std::span<const std::uint32_t> token_ids,
    std::span<const float> raw_scores, std::uint32_t vocabulary_size,
    std::span<const std::uint16_t> token_to_experts,
    std::span<DeepSeekExpertRoute> routes) {
  const auto token_count = static_cast<std::uint32_t>(token_ids.size());
  if (token_count == 0 || vocabulary_size == 0 ||
      raw_scores.size() != static_cast<std::size_t>(token_count) *
                               DeepSeekExpertSubwavePlan::kExpertCount ||
      token_to_experts.size() !=
          static_cast<std::size_t>(vocabulary_size) *
              DeepSeekExpertSubwavePlan::kRoutesPerToken ||
      routes.size() != static_cast<std::size_t>(token_count) *
                           DeepSeekExpertSubwavePlan::kRoutesPerToken) {
    return Status::InvalidArgument("DeepSeek hash router tensor shape is invalid");
  }
  for (std::uint32_t token = 0; token < token_count; ++token) {
    if (token_ids[token] >= vocabulary_size) {
      return Status::InvalidArgument("DeepSeek hash router token is out of range");
    }
    std::array<bool, DeepSeekExpertSubwavePlan::kExpertCount> seen{};
    std::array<float, DeepSeekExpertSubwavePlan::kRoutesPerToken> bases{};
    float denominator = 0.0F;
    const auto table_offset =
        static_cast<std::size_t>(token_ids[token]) *
        DeepSeekExpertSubwavePlan::kRoutesPerToken;
    for (std::uint32_t ordinal = 0;
         ordinal < DeepSeekExpertSubwavePlan::kRoutesPerToken; ++ordinal) {
      const auto expert = token_to_experts[table_offset + ordinal];
      if (expert >= DeepSeekExpertSubwavePlan::kExpertCount || seen[expert]) {
        return Status::InvalidArgument(
            "DeepSeek hash router table or score is invalid");
      }
      const auto raw = raw_scores[static_cast<std::size_t>(token) *
                                      DeepSeekExpertSubwavePlan::kExpertCount +
                                  expert];
      if (!std::isfinite(raw)) {
        return Status::InvalidArgument(
            "DeepSeek hash router table or score is invalid");
      }
      seen[expert] = true;
      bases[ordinal] = hash_router_base(raw);
      denominator += bases[ordinal];
    }
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
      return Status::Internal("DeepSeek hash router weight sum is invalid");
    }
    const auto output_offset = static_cast<std::size_t>(token) *
                               DeepSeekExpertSubwavePlan::kRoutesPerToken;
    for (std::uint32_t ordinal = 0;
         ordinal < DeepSeekExpertSubwavePlan::kRoutesPerToken; ++ordinal) {
      routes[output_offset + ordinal] = {
          token_to_experts[table_offset + ordinal], token,
          static_cast<std::uint8_t>(ordinal),
          bases[ordinal] / denominator *
              DeepSeekHashRouter::kRoutedScalingFactor};
    }
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekExpertSubwavePlan> DeepSeekHashRouter::Route(
    const std::vector<std::uint32_t>& token_ids,
    const std::vector<float>& raw_scores, std::uint32_t vocabulary_size,
    const std::vector<std::uint16_t>& token_to_experts) {
  std::vector<DeepSeekExpertRoute> routes(
      token_ids.size() * DeepSeekExpertSubwavePlan::kRoutesPerToken);
  const auto status = route_into(token_ids, raw_scores, vocabulary_size,
                                 token_to_experts, routes);
  if (!status.ok()) return status;
  return DeepSeekExpertSubwavePlan::Create(
      static_cast<std::uint32_t>(token_ids.size()), std::move(routes));
}

Status DeepSeekHashRouter::RouteInto(
    std::uint32_t layer, std::span<const std::uint32_t> token_ids,
    std::span<const float> raw_scores, std::uint32_t vocabulary_size,
    std::span<const std::uint16_t> token_to_experts,
    DeepSeekRouteScratchArena& scratch,
    DeepSeekExpertPlanStore& store) {
  auto routes = scratch.routes(static_cast<std::uint32_t>(token_ids.size()));
  if (!routes.ok()) return routes.status();
  const auto status = route_into(token_ids, raw_scores, vocabulary_size,
                                 token_to_experts, *routes);
  if (!status.ok()) return status;
  return store.publish_routes(
      layer, static_cast<std::uint32_t>(token_ids.size()), *routes);
}

}  // namespace pih
