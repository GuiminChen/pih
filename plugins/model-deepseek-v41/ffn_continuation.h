#pragma once
#include "ffn_route.h"
#include "expert_batch.h"
#include "expert_pipeline.h"
#include "expert_workspace_owner.h"

namespace pih::deepseek_v41 {
Status ValidateFfnContinuation(const FlashConfig& config, const FfnRouteLaunch& route,
    const ExpertCounts& counts, std::span<const ExpertTokenChainLaunch> batch,
    const ExpertResidualLaunch& tail);
enum class FfnContinuationState { kWaitingBatch, kWaitingTail, kWaitingRetirement, kComplete, kFailed };
// Continues an already observed route operation. Borrowed GPU resources remain
// live through both stages. This multi-rank entry does not own or retry ranks.
class FfnContinuation final {
 public:
  using Clock = std::chrono::steady_clock;
  FfnContinuation(const FfnContinuation&) = delete;
  FfnContinuation& operator=(const FfnContinuation&) = delete;
  FfnContinuation(FfnContinuation&& other) noexcept;
  FfnContinuation& operator=(FfnContinuation&&) = delete;
  static Result<FfnContinuation> Start(const FlashConfig& config, const FfnRouteLaunch& route,
      const ExpertCounts& counts, std::span<const ExpertWeights> weights, ExpertWorkspaceOwner& workspace,
      const ExpertResidualLaunch& tail, std::uintptr_t communicator,
      const EngramCompletionResources& resources, Clock::time_point deadline, std::uint64_t reservation = 0);
  Result<FfnContinuationState> Advance();
 private:
  FfnContinuation() = default;
  ExpertResidualLaunch tail_launch_{};
  std::uint32_t world_ = 0, rank_ = 0;
  std::uintptr_t communicator_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<ExpertBatch> batch_;
  std::optional<ExpertTensorParallel> tail_;
  ExpertWorkspaceOwner* workspace_ = nullptr;
  std::uint64_t reservation_ = 0;
  FfnContinuationState state_ = FfnContinuationState::kFailed;
};
}  // namespace pih::deepseek_v41
