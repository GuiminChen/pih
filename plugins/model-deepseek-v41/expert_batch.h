#pragma once
#include "expert_chain.h"
#include "expert_counts.h"
#include <span>

namespace pih::deepseek_v41 {
Status ValidateExpertBatch(std::span<const ExpertTokenChainLaunch> experts, const ExpertCounts& counts);
class ExpertBatch final {
 public:
  using Clock = std::chrono::steady_clock;
  ExpertBatch(const ExpertBatch&) = delete;
  ExpertBatch& operator=(const ExpertBatch&) = delete;
  ExpertBatch(ExpertBatch&& other) noexcept;
  ExpertBatch& operator=(ExpertBatch&&) = delete;
  // All local experts in ascending global ID, including zero-row experts.
  // Initializes accumulator once, executes each chain, records completion.
  // Borrows all GPU resources; failure does not cancel already submitted work.
  static Result<ExpertBatch> Start(std::span<const ExpertTokenChainLaunch> experts,
      const ExpertCounts& counts, const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<bool> Poll();
 private:
  ExpertBatch() = default;
  std::optional<EngramCompletion> completion_;
  Clock::time_point deadline_{};
  bool failed_ = true, complete_ = false;
};
}  // namespace pih::deepseek_v41
