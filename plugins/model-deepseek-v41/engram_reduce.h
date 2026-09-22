#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
enum class EngramReductionState { kPending, kEnqueued, kFailed };
struct Bf16ReductionLaunch final {
  EngramDeviceRegion output;
  std::uintptr_t stream = 0;
  std::uint32_t world_size = 0, rank = 0;
};
Status ValidateBf16Reduction(const Bf16ReductionLaunch& launch, std::uintptr_t communicator);
struct Fp32ReductionLaunch final {
  EngramDeviceRegion output;
  std::uintptr_t stream = 0;
  std::uint32_t world_size = 0, rank = 0;
};
Status ValidateFp32Reduction(const Fp32ReductionLaunch& launch, std::uintptr_t communicator);
struct Fp32GatherLaunch final {
  EngramDeviceRegion input, output;
  std::uintptr_t stream = 0;
  std::uint32_t world_size = 0, rank = 0;
};
// Out-of-place, contiguous rank-major gather of equal-sized FP32 inputs.
Status ValidateFp32Gather(const Fp32GatherLaunch& launch, std::uintptr_t communicator);
// Read-only preflight, only while the caller exclusively owns an idle
// communicator. Does not authenticate distributed agreement or buffer lifetimes.
Status ValidateEngramReduction(const EngramLookupLaunch& lookup, std::uintptr_t communicator);
// Borrowed communicator/buffers: their owner must serialize use, maintain their
// lifetime and enforce a deadline/abort policy. No communicator is created here.
// Submit after the corresponding producer enqueue on the same explicit stream.
// The completion-state wrapper serves both SUM reductions and FP32 all-gather.
class EngramReduction final {
 public:
  EngramReduction(const EngramReduction&) = delete;
  EngramReduction& operator=(const EngramReduction&) = delete;
  EngramReduction(EngramReduction&& other) noexcept;
  EngramReduction& operator=(EngramReduction&&) = delete;
  static Result<EngramReduction> Submit(const EngramLookupLaunch& lookup,
      std::uintptr_t communicator);
  static Result<EngramReduction> SubmitBf16(const Bf16ReductionLaunch& launch,
      std::uintptr_t communicator);
  static Result<EngramReduction> SubmitFp32(const Fp32ReductionLaunch& launch,
      std::uintptr_t communicator);
  static Result<EngramReduction> SubmitGatherFp32(const Fp32GatherLaunch& launch,
      std::uintptr_t communicator);
  // Never launch projection or any subsequent NCCL/CUDA work on the associated
  // stream until this returns kEnqueued. That state is NOT GPU completion.
  Result<EngramReductionState> PollEnqueued();
  EngramReductionState state() const noexcept { return state_; }
 private:
  EngramReduction() = default;
  std::uintptr_t communicator_ = 0;
  EngramReductionState state_ = EngramReductionState::kFailed;
};
}  // namespace pih::deepseek_v41
