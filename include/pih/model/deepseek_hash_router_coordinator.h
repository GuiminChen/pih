#pragma once

#include <span>

#include "pih/model/deepseek_hash_router.h"
#include "pih/model/deepseek_learned_router_coordinator.h"

namespace pih {

struct DeepSeekHashRouterSubmission final {
  DeepSeekLearnedRouterSubmission projection;
  std::span<const std::uint32_t> token_ids;
  std::uint32_t vocabulary_size = 0;
  std::span<const std::uint16_t> token_to_experts;
};

class DeepSeekHashRouterCoordinator final {
 public:
  static Result<DeepSeekHashRouterCoordinator> Create(
      std::uint32_t maximum_tokens, DeepSeekRouteScratchArena& scratch,
      DeepSeekExpertPlanStore& store,
      DeepSeekLearnedRouterOperations& operations);
  Status launch(const DeepSeekHashRouterSubmission& submission);
  Result<DeepSeekStageComputeStatus> poll();
  [[nodiscard]] DeepSeekExpertPlanProvider* plan_provider() const noexcept {
    return store_;
  }

 private:
  Status poison(Status status);
  std::uint32_t maximum_tokens_ = 0;
  DeepSeekRouteScratchArena* scratch_ = nullptr;
  DeepSeekExpertPlanStore* store_ = nullptr;
  DeepSeekLearnedRouterOperations* operations_ = nullptr;
  DeepSeekHashRouterSubmission active_;
  bool inflight_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
