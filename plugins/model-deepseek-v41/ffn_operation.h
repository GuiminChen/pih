#pragma once
#include "ffn_continuation.h"

namespace pih::deepseek_v41 {
enum class FfnOperationState { kWaitingCounts, kWaitingExecution, kComplete, kFailed };
// Owns host descriptors, borrows admitted device weights, buffers, workspace,
// communicator and completion resources. One instance is one backbone FFN step.
class FfnOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  FfnOperation(const FfnOperation&) = delete;
  FfnOperation& operator=(const FfnOperation&) = delete;
  FfnOperation(FfnOperation&& other) noexcept;
  FfnOperation& operator=(FfnOperation&&) = delete;
  static Result<FfnOperation> Start(const FlashConfig& config, const FfnRouteLaunch& route,
      std::span<const ExpertWeights> weights, ExpertWorkspaceOwner& workspace,
      const ExpertResidualLaunch& tail_template, EngramDeviceRegion host_counts,
      std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline,
      std::uint64_t reservation = 0);
  Result<FfnOperationState> Advance();
 private:
  FfnOperation() = default;
  FlashConfig config_{};
  FfnRouteLaunch route_{};
  ExpertResidualLaunch tail_{};
  std::vector<ExpertWeights> weights_;
  ExpertWorkspaceOwner* workspace_ = nullptr;
  std::uint64_t reservation_ = 0;
  std::uintptr_t communicator_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<ExpertCounts> counts_;
  std::optional<FfnContinuation> execution_;
  FfnOperationState state_ = FfnOperationState::kFailed;
};
}  // namespace pih::deepseek_v41
