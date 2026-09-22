#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_expert_subwave_plan.h"

namespace pih {

class DeepSeekRouteScratchArena final {
 public:
  static Result<DeepSeekRouteScratchArena> Create(
      std::uint32_t maximum_token_count);
  Result<std::span<DeepSeekExpertRoute>> routes(
      std::uint32_t token_count);
  [[nodiscard]] std::uint32_t maximum_token_count() const noexcept {
    return maximum_token_count_;
  }

 private:
  std::uint32_t maximum_token_count_ = 0;
  std::vector<DeepSeekExpertRoute> routes_;
};

}  // namespace pih
