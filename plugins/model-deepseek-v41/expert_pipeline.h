#pragma once
#include "expert_residual.h"
#include "engram_reduce.h"
#include "engram_completion.h"
#include <chrono>
#include <optional>

namespace pih::deepseek_v41 {
enum class ExpertPipelineState { kWaitingReduction, kWaitingCompletion, kComplete, kFailed };
// Starts after rank-local expert accumulation is ready on the stream. Does
// not submit that batch or prove that every rank used the same token ordering.
class ExpertTensorParallel final {
 public:
  using Clock = std::chrono::steady_clock;
  ExpertTensorParallel(const ExpertTensorParallel&) = delete;
  ExpertTensorParallel& operator=(const ExpertTensorParallel&) = delete;
  ExpertTensorParallel(ExpertTensorParallel&& other) noexcept;
  ExpertTensorParallel& operator=(ExpertTensorParallel&&) = delete;
  static Result<ExpertTensorParallel> Start(const ExpertResidualLaunch& launch,
      std::uint32_t world_size, std::uint32_t rank, std::uintptr_t communicator,
      const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<ExpertPipelineState> Advance();
 private:
  ExpertTensorParallel() = default;
  ExpertResidualLaunch launch_{};
  Fp32ReductionLaunch reduction_launch_{};
  std::uintptr_t communicator_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<EngramReduction> reduction_;
  std::optional<EngramCompletion> completion_;
  ExpertPipelineState state_ = ExpertPipelineState::kFailed;
};
}  // namespace pih::deepseek_v41
