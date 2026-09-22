#pragma once
#include "engram_reduce.h"
#include "engram_completion.h"
#include <chrono>
#include <optional>

namespace pih::deepseek_v41 {
enum class EngramPipelineState { kWaitingReduction, kWaitingCompletion, kComplete, kFailed };
// One rank's borrowed-resource operation. Not a distributed supervisor and not
// thread safe. The owner supplies an initialized nonblocking communicator,
// matching rank schedules, resource lifetimes and generation-wide abort policy.
class EngramTensorParallel final {
 public:
  using Clock = std::chrono::steady_clock;
  EngramTensorParallel(const EngramTensorParallel&) = delete;
  EngramTensorParallel& operator=(const EngramTensorParallel&) = delete;
  EngramTensorParallel(EngramTensorParallel&& other) noexcept;
  EngramTensorParallel& operator=(EngramTensorParallel&&) = delete;
  static Result<EngramTensorParallel> Start(const EngramLaunch& launch,
      std::uintptr_t communicator, const EngramCompletionResources& completion,
      Clock::time_point deadline);
  // Non-waiting poll: pending returns immediately. Projection/gate are submitted
  // exactly once after NCCL confirms enqueue. Only kComplete permits output use.
  Result<EngramPipelineState> Advance();
  EngramPipelineState state() const noexcept { return state_; }
 private:
  EngramTensorParallel() = default;
  EngramLaunch launch_{};
  std::uintptr_t communicator_ = 0;
  Clock::time_point deadline_{};
  std::optional<EngramReduction> reduction_;
  EngramCompletionResources completion_resources_{};
  std::optional<EngramCompletion> completion_;
  EngramPipelineState state_ = EngramPipelineState::kFailed;
};
}  // namespace pih::deepseek_v41
