#include "pih/model/deepseek_route_scratch_arena.h"

#include <limits>

namespace pih {

Result<DeepSeekRouteScratchArena> DeepSeekRouteScratchArena::Create(
    std::uint32_t maximum_token_count) {
  if (maximum_token_count == 0 ||
      maximum_token_count > std::numeric_limits<std::uint32_t>::max() /
                                DeepSeekExpertSubwavePlan::kRoutesPerToken) {
    return Status::InvalidArgument("DeepSeek route scratch capacity is invalid");
  }
  DeepSeekRouteScratchArena arena;
  arena.maximum_token_count_ = maximum_token_count;
  arena.routes_.resize(static_cast<std::size_t>(maximum_token_count) *
                       DeepSeekExpertSubwavePlan::kRoutesPerToken);
  return arena;
}

Result<std::span<DeepSeekExpertRoute>> DeepSeekRouteScratchArena::routes(
    std::uint32_t token_count) {
  if (token_count == 0 || token_count > maximum_token_count_) {
    return Status::ResourceExhausted(
        "DeepSeek route scratch cannot represent token count");
  }
  return std::span(routes_).first(
      static_cast<std::size_t>(token_count) *
      DeepSeekExpertSubwavePlan::kRoutesPerToken);
}

}  // namespace pih
