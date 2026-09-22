#include "pih/model/deepseek_expert_subwave_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pih {

Result<DeepSeekExpertSubwavePlan> DeepSeekExpertSubwavePlan::Reserve(
    std::uint32_t maximum_token_count) {
  if (maximum_token_count == 0 ||
      maximum_token_count > std::numeric_limits<std::uint32_t>::max() /
                                kRoutesPerToken) {
    return Status::InvalidArgument("DeepSeek expert route capacity is invalid");
  }
  DeepSeekExpertSubwavePlan plan;
  plan.maximum_token_count_ = maximum_token_count;
  plan.routes_.reserve(static_cast<std::size_t>(maximum_token_count) *
                       kRoutesPerToken);
  return plan;
}

Result<DeepSeekExpertSubwavePlan> DeepSeekExpertSubwavePlan::Create(
    std::uint32_t token_count, std::vector<DeepSeekExpertRoute> routes) {
  auto plan = Reserve(token_count);
  if (!plan.ok()) return plan.status();
  const auto status = plan->materialize(token_count, routes);
  if (!status.ok()) return status;
  return plan;
}

Status DeepSeekExpertSubwavePlan::materialize(
    std::uint32_t token_count,
    std::span<const DeepSeekExpertRoute> routes) {
  if (token_count == 0 || token_count > maximum_token_count_ ||
      routes.size() != static_cast<std::size_t>(token_count) *
                           kRoutesPerToken) {
    return Status::InvalidArgument(
        "DeepSeek expert route table must contain exactly six routes per token");
  }
  token_count_ = 0;
  workspace_bytes_ = 0;
  expert_offsets_.fill(0);
  routes_.assign(routes.begin(), routes.end());
  for (const auto& route : routes_) {
    if (route.expert_id >= kExpertCount || route.token_index >= token_count ||
        route.route_ordinal >= kRoutesPerToken ||
        !std::isfinite(route.weight) || route.weight < 0.0F) {
      return Status::InvalidArgument("DeepSeek expert route is invalid");
    }
  }

  std::sort(routes_.begin(), routes_.end(),
            [](const DeepSeekExpertRoute& left,
               const DeepSeekExpertRoute& right) {
              if (left.token_index != right.token_index) {
                return left.token_index < right.token_index;
              }
              return left.route_ordinal < right.route_ordinal;
            });
  for (std::uint32_t token = 0; token < token_count; ++token) {
    const auto begin = static_cast<std::size_t>(token) * kRoutesPerToken;
    for (std::uint32_t ordinal = 0; ordinal < kRoutesPerToken; ++ordinal) {
      const auto& route = routes_[begin + ordinal];
      if (route.token_index != token || route.route_ordinal != ordinal) {
        return Status::InvalidArgument(
            "DeepSeek token route ordinal set is incomplete");
      }
      for (std::uint32_t previous = 0; previous < ordinal; ++previous) {
        if (routes_[begin + previous].expert_id == route.expert_id) {
          return Status::InvalidArgument(
              "DeepSeek token contains a duplicate expert");
        }
      }
    }
  }

  std::sort(routes_.begin(), routes_.end(),
            [](const DeepSeekExpertRoute& left,
               const DeepSeekExpertRoute& right) {
              if (left.expert_id != right.expert_id) {
                return left.expert_id < right.expert_id;
              }
              if (left.token_index != right.token_index) {
                return left.token_index < right.token_index;
              }
              return left.route_ordinal < right.route_ordinal;
            });

  token_count_ = token_count;
  std::size_t cursor = 0;
  for (std::uint32_t expert = 0; expert < kExpertCount; ++expert) {
    expert_offsets_[expert] = static_cast<std::uint32_t>(cursor);
    while (cursor < routes_.size() && routes_[cursor].expert_id == expert) {
      ++cursor;
    }
  }
  expert_offsets_[kExpertCount] = static_cast<std::uint32_t>(cursor);
  workspace_bytes_ = std::uint64_t{16480} * token_count + 1280U;
  return Status::Ok();
}

}  // namespace pih
