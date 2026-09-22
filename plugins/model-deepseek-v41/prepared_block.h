#pragma once
#include "attention_pipeline.h"
#include "ffn_operation.h"

namespace pih::deepseek_v41 {
struct PreparedBlockLaunch final {
  AttentionResidualLaunch attention;
  // Prepared by this layer's attention mHC input stage, carried into FFN.
  EngramDeviceRegion attention_pre;
  FfnRouteLaunch ffn;
  ExpertResidualLaunch tail;
};
enum class PreparedBlockState { kWaitingAttention, kWaitingFfn, kComplete, kFailed };
// Requires query/window/compressed/indexed KV and attention coefficients already
// prepared. Does not own/cache/compute those upstream inputs.
class PreparedBlockOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  PreparedBlockOperation(const PreparedBlockOperation&) = delete;
  PreparedBlockOperation& operator=(const PreparedBlockOperation&) = delete;
  PreparedBlockOperation(PreparedBlockOperation&& other) noexcept;
  PreparedBlockOperation& operator=(PreparedBlockOperation&&) = delete;
  static Result<PreparedBlockOperation> Start(const FlashConfig& config, const PreparedBlockLaunch& launch,
      std::span<const ExpertWeights> weights, ExpertWorkspaceOwner& workspace,
      EngramDeviceRegion host_counts, std::uintptr_t communicator,
      const EngramCompletionResources& resources, Clock::time_point deadline, std::uint64_t reservation = 0);
  Result<PreparedBlockState> Advance();
 private:
  PreparedBlockOperation() = default;
  FlashConfig config_{};
  PreparedBlockLaunch launch_{};
  std::vector<ExpertWeights> weights_;
  ExpertWorkspaceOwner* workspace_ = nullptr;
  EngramDeviceRegion host_counts_{};
  std::uintptr_t communicator_ = 0;
  std::uint64_t reservation_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<AttentionTensorParallel> attention_;
  std::optional<FfnOperation> ffn_;
  PreparedBlockState state_ = PreparedBlockState::kFailed;
};
}  // namespace pih::deepseek_v41
