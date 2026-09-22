#pragma once
#include "attention_reduce.h"
#include "attention_residual.h"
#include "engram_completion.h"
#include <chrono>
#include <optional>

namespace pih::deepseek_v41 {
enum class AttentionPipelineState { kWaitingReduction, kWaitingCompletion, kComplete, kFailed };
// Borrowed-resource output-to-residual operation. Query/cache/coefficients must
// already be ready on the stream; this is not a full attention input scheduler.
class AttentionTensorParallel final {
 public:
  using Clock = std::chrono::steady_clock;
  AttentionTensorParallel(const AttentionTensorParallel&) = delete;
  AttentionTensorParallel& operator=(const AttentionTensorParallel&) = delete;
  AttentionTensorParallel(AttentionTensorParallel&& other) noexcept;
  AttentionTensorParallel& operator=(AttentionTensorParallel&&) = delete;
  static Result<AttentionTensorParallel> Start(const AttentionResidualLaunch& launch,
      std::uint32_t rank, std::uintptr_t communicator, const EngramCompletionResources& resources,
      Clock::time_point deadline);
  Result<AttentionPipelineState> Advance();
  AttentionPipelineState state() const noexcept { return state_; }
 private:
  AttentionTensorParallel() = default;
  AttentionResidualLaunch launch_{};
  std::uint32_t rank_ = 0;
  std::uintptr_t communicator_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<EngramReduction> reduction_;
  std::optional<EngramCompletion> completion_;
  AttentionPipelineState state_ = AttentionPipelineState::kFailed;
};
}  // namespace pih::deepseek_v41
