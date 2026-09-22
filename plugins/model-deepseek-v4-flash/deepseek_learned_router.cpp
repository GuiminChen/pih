#include "pih/model/deepseek_learned_router.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace pih {
namespace {

float stable_softplus(float value) {
  return value > 0.0F ? value + std::log1p(std::exp(-value))
                      : std::log1p(std::exp(value));
}

struct RankedExpert final {
  std::uint16_t expert = 0;
  float base = 0.0F;
  float selection = 0.0F;
};

Status route_into(
    std::uint32_t token_count, std::span<const float> raw_scores,
    const std::array<float, DeepSeekExpertSubwavePlan::kExpertCount>& bias,
    std::span<DeepSeekExpertRoute> routes) {
  if (token_count == 0 ||
      raw_scores.size() != static_cast<std::size_t>(token_count) *
                               DeepSeekExpertSubwavePlan::kExpertCount ||
      routes.size() != static_cast<std::size_t>(token_count) *
                           DeepSeekExpertSubwavePlan::kRoutesPerToken) {
    return Status::InvalidArgument(
        "DeepSeek learned router score shape is invalid");
  }
  if (!std::all_of(bias.begin(), bias.end(),
                   [](float value) { return std::isfinite(value); })) {
    return Status::InvalidArgument("DeepSeek learned router bias is non-finite");
  }
  for (std::uint32_t token = 0; token < token_count; ++token) {
    std::array<RankedExpert, DeepSeekExpertSubwavePlan::kExpertCount> ranked;
    for (std::uint16_t expert = 0;
         expert < DeepSeekExpertSubwavePlan::kExpertCount; ++expert) {
      const auto raw = raw_scores[static_cast<std::size_t>(token) *
                                      DeepSeekExpertSubwavePlan::kExpertCount +
                                  expert];
      if (!std::isfinite(raw)) {
        return Status::InvalidArgument(
            "DeepSeek learned router raw score is non-finite");
      }
      const auto base = std::sqrt(stable_softplus(raw));
      ranked[expert] = {expert, base, base + bias[expert]};
    }
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + DeepSeekExpertSubwavePlan::kRoutesPerToken,
        ranked.end(), [](const RankedExpert& left, const RankedExpert& right) {
          if (left.selection != right.selection) {
            return left.selection > right.selection;
          }
          return left.expert < right.expert;
        });
    float denominator = 0.0F;
    for (std::uint32_t ordinal = 0;
         ordinal < DeepSeekExpertSubwavePlan::kRoutesPerToken; ++ordinal) {
      denominator += ranked[ordinal].base;
    }
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
      return Status::Internal("DeepSeek learned router weight sum is invalid");
    }
    const auto begin = static_cast<std::size_t>(token) *
                       DeepSeekExpertSubwavePlan::kRoutesPerToken;
    for (std::uint32_t ordinal = 0;
         ordinal < DeepSeekExpertSubwavePlan::kRoutesPerToken; ++ordinal) {
      routes[begin + ordinal] = {
          ranked[ordinal].expert, token, static_cast<std::uint8_t>(ordinal),
          ranked[ordinal].base / denominator *
              DeepSeekLearnedRouter::kRoutedScalingFactor};
    }
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekExpertSubwavePlan> DeepSeekLearnedRouter::Route(
    std::uint32_t token_count, const std::vector<float>& raw_scores,
    const std::array<float, DeepSeekExpertSubwavePlan::kExpertCount>& bias) {
  std::vector<DeepSeekExpertRoute> routes(
      static_cast<std::size_t>(token_count) *
      DeepSeekExpertSubwavePlan::kRoutesPerToken);
  const auto status = route_into(token_count, raw_scores, bias, routes);
  if (!status.ok()) return status;
  return DeepSeekExpertSubwavePlan::Create(token_count, std::move(routes));
}

Status DeepSeekLearnedRouter::RouteInto(
    std::uint32_t layer, std::uint32_t token_count,
    std::span<const float> raw_scores,
    const std::array<float, DeepSeekExpertSubwavePlan::kExpertCount>& bias,
    DeepSeekRouteScratchArena& scratch,
    DeepSeekExpertPlanStore& store) {
  auto routes = scratch.routes(token_count);
  if (!routes.ok()) return routes.status();
  const auto status = route_into(token_count, raw_scores, bias, *routes);
  if (!status.ok()) return status;
  return store.publish_routes(layer, token_count, *routes);
}

}  // namespace pih
