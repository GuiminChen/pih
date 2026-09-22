#pragma once

#include <cstdint>
#include <string_view>

namespace pih {

inline constexpr std::string_view kDeepSeekExecutionTopologyAbi =
    "pih_deepseek_execution_topology_v1";
inline constexpr std::string_view kDeepSeekInProcessRankSetAbi =
    "in_process_rank_set_development_v1";
inline constexpr std::string_view kDeepSeekOneProcessPerRankAbi =
    "one_process_per_rank_v1";

enum class DeepSeekExecutionTopology : std::uint8_t {
  kInProcessRankSetDevelopment = 1,
  kOneProcessPerRank = 2,
};

[[nodiscard]] constexpr std::string_view deepseek_execution_topology_name(
    DeepSeekExecutionTopology topology) noexcept {
  switch (topology) {
    case DeepSeekExecutionTopology::kInProcessRankSetDevelopment:
      return kDeepSeekInProcessRankSetAbi;
    case DeepSeekExecutionTopology::kOneProcessPerRank:
      return kDeepSeekOneProcessPerRankAbi;
  }
  return {};
}

}  // namespace pih
