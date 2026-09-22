#pragma once
#include "block_sequence.h"
#include "model_head.h"
#include "uploaded_bindings.h"

namespace pih::deepseek_v41 {
enum class HeadOperationState { kWaitingGather, kWaitingCompletion, kComplete, kFailed };
// Ordinary next-token logits on every rank, not sampling or full-prefill logits.
// Borrows sequence, weight/scratch, communicator, completion resources.
class HeadOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  HeadOperation(const HeadOperation&) = delete;
  HeadOperation& operator=(const HeadOperation&) = delete;
  HeadOperation(HeadOperation&& other) noexcept;
  HeadOperation& operator=(HeadOperation&&) = delete;
  ~HeadOperation();
  static Result<HeadOperation> Start(BlockSequence& sequence, ModelHeadLaunch launch, const BackboneWeightUpload& weights,
      EngramDeviceRegion logits, std::uintptr_t communicator,
      const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<HeadOperationState> Advance();
  Result<EngramDeviceRegion> Logits() const;
 private:
  friend class SamplingOperation;
  HeadOperation() = default;
  Result<HeadOperationState> AdvanceImpl();
  BlockSequence* sequence_ = nullptr;
  EngramDeviceRegion logits_{}, error_{};
  std::uintptr_t stream_ = 0;
  std::uint32_t step_end_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<EngramReduction> gather_;
  std::optional<EngramCompletion> completion_;
  HeadOperationState state_ = HeadOperationState::kFailed;
};
}  // namespace pih::deepseek_v41
